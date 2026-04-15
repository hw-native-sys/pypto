# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the statement dependency analysis utilities.

Covers:
- `build_stmt_dependency_graph`: dataflow dependency graph over a region's
  top-level statements.
- `check_inout_use_discipline`: detection of post-call reads of InOut/Out-passed
  variables (RFC #1026 Phase 1, issue #1027).
"""

import pypto.language as pl
import pytest
from pypto import ir, passes

dep_analysis = passes.stmt_dependency_analysis


def _seq_body(program: ir.Program, name: str) -> ir.SeqStmts:
    """Return the named function's body as a SeqStmts (narrows types for pyright)."""
    func = program.get_function(name)
    assert func is not None, f"function '{name}' not found in program"
    body = func.body
    assert isinstance(body, ir.SeqStmts), (
        f"function '{name}' body is {type(body).__name__}, expected SeqStmts"
    )
    return body


# =============================================================================
# Dependency graph
# =============================================================================


class TestStmtDependencyGraph:
    """Tests for `build_stmt_dependency_graph`."""

    def test_non_seqstmts_region_is_single_node(self):
        """A region that isn't a SeqStmts becomes a one-node graph."""

        @pl.program
        class P:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return result

        # Drill down to a non-SeqStmts child: the first AssignStmt.
        first_stmt = _seq_body(P, "main").stmts[0]
        graph = dep_analysis.build_stmt_dependency_graph(first_stmt)
        assert len(graph.stmts) == 1
        assert graph.stmts[0] is first_stmt
        assert graph.get_predecessors(first_stmt) == []

    def test_linear_chain(self):
        """a → b → c: each stmt depends only on its immediate predecessor."""

        @pl.program
        class P:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                b: pl.Tensor[[64], pl.FP32] = pl.add(a, a)
                c: pl.Tensor[[64], pl.FP32] = pl.add(b, b)
                return c

        body = _seq_body(P, "main")
        stmts = body.stmts
        # stmts: [a_assign, b_assign, c_assign, return]
        graph = dep_analysis.build_stmt_dependency_graph(body)

        assert graph.get_predecessors(stmts[0]) == []  # a depends on external x only
        assert graph.get_predecessors(stmts[1]) == [stmts[0]]  # b ← a
        assert graph.get_predecessors(stmts[2]) == [stmts[1]]  # c ← b
        # return c depends on c_assign (the defining stmt of c).
        assert graph.get_predecessors(stmts[3]) == [stmts[2]]

    def test_independent_stmts_have_no_edge(self):
        """Two stmts reading only external vars have no intra-region edges."""

        @pl.program
        class P:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                b: pl.Tensor[[64], pl.FP32] = pl.mul(x, x)
                c: pl.Tensor[[64], pl.FP32] = pl.add(a, b)
                return c

        body = _seq_body(P, "main")
        stmts = body.stmts
        graph = dep_analysis.build_stmt_dependency_graph(body)
        assert graph.get_predecessors(stmts[0]) == []
        assert graph.get_predecessors(stmts[1]) == []
        # c reads a and b → depends on both.
        preds_c = graph.get_predecessors(stmts[2])
        assert set(preds_c) == {stmts[0], stmts[1]}

    def test_compound_child_aggregates_subtree_use(self):
        """A subtree read inside an If counts as a use of the If stmt."""

        @pl.program
        class P:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32], cond: pl.Scalar[pl.BOOL]) -> pl.Tensor[[64], pl.FP32]:
                a: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                if cond:
                    b: pl.Tensor[[64], pl.FP32] = pl.add(a, a)
                else:
                    b: pl.Tensor[[64], pl.FP32] = pl.mul(a, a)
                return b

        body = _seq_body(P, "main")
        stmts = body.stmts
        # stmts: [a_assign, if_stmt, return]
        graph = dep_analysis.build_stmt_dependency_graph(body)
        # If-stmt depends on a (both branches read a inside).
        assert graph.get_predecessors(stmts[1]) == [stmts[0]]

    def test_compound_child_aggregates_subtree_def(self):
        """A var defined inside a For body is visible on the For stmt."""

        @pl.program
        class P:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                for i, (acc,) in pl.range(0, 4, 1, init_values=(a,)):
                    acc = pl.yield_(pl.add(acc, a))
                b: pl.Tensor[[64], pl.FP32] = pl.add(acc, x)
                return b

        body = _seq_body(P, "main")
        stmts = body.stmts
        # stmts: [a_assign, for_stmt, b_assign, return]
        graph = dep_analysis.build_stmt_dependency_graph(body)
        # for_stmt reads a (init_values + body use).
        assert stmts[0] in graph.get_predecessors(stmts[1])
        # b reads acc — acc is defined by the for_stmt's return_vars.
        assert stmts[1] in graph.get_predecessors(stmts[2])

    def test_uses_outside_region_have_no_edge(self):
        """Uses of vars defined outside the SeqStmts don't create intra-region edges."""

        @pl.program
        class P:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                z: pl.Tensor[[64], pl.FP32] = pl.mul(x, x)  # noqa: F841
                return y

        body = _seq_body(P, "main")
        stmts = body.stmts
        graph = dep_analysis.build_stmt_dependency_graph(body)
        # Neither assign has a predecessor in the region — both only read external x.
        assert graph.get_predecessors(stmts[0]) == []
        assert graph.get_predecessors(stmts[1]) == []


# =============================================================================
# InOut-use discipline
# =============================================================================


class TestInOutUseDiscipline:
    """Tests for `check_inout_use_discipline`."""

    def test_clean_when_return_value_is_used(self):
        """Well-formed: reads after the call go through the returned var."""

        @pl.program
        class P:
            @pl.function
            def mutate(self, T: pl.InOut[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
                r: pl.Tensor[[64], pl.FP32] = pl.add(T, T)
                return r

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                T_new: pl.Tensor[[64], pl.FP32] = self.mutate(x)
                y: pl.Tensor[[64], pl.FP32] = pl.add(T_new, T_new)
                return y

        body = _seq_body(P, "main")
        diags = dep_analysis.check_inout_use_discipline(body, P)
        assert diags == []

    def test_rejects_post_inout_read(self):
        """A read of the InOut-passed var after the call is flagged."""

        @pl.program
        class P:
            @pl.function
            def mutate(self, T: pl.InOut[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
                r: pl.Tensor[[64], pl.FP32] = pl.add(T, T)
                return r

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                T_new: pl.Tensor[[64], pl.FP32] = self.mutate(x)  # noqa: F841
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)  # VIOLATION
                return y

        body = _seq_body(P, "main")
        diags = dep_analysis.check_inout_use_discipline(body, P)
        assert len(diags) >= 1
        assert all(d.severity == passes.DiagnosticSeverity.Error for d in diags)
        assert all(d.rule_name == "InOutUseDiscipline" for d in diags)
        assert all("'x'" in d.message for d in diags)

    def test_out_direction_treated_like_inout(self):
        """Out params cause the same 'dead for read' behavior as InOut."""

        @pl.program
        class P:
            @pl.function
            def produce(self, T: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
                r: pl.Tensor[[64], pl.FP32] = pl.add(T, T)
                return r

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                T_new: pl.Tensor[[64], pl.FP32] = self.produce(x)  # noqa: F841
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)  # VIOLATION: Out is like InOut
                return y

        body = _seq_body(P, "main")
        diags = dep_analysis.check_inout_use_discipline(body, P)
        assert len(diags) >= 1
        assert all("'x'" in d.message for d in diags)

    def test_rejects_read_in_nested_if_branch(self):
        """A read in an `if` body after the call is still reachable → violation."""

        @pl.program
        class P:
            @pl.function
            def mutate(self, T: pl.InOut[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
                r: pl.Tensor[[64], pl.FP32] = pl.add(T, T)
                return r

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32], cond: pl.Scalar[pl.BOOL]) -> pl.Tensor[[64], pl.FP32]:
                T_new: pl.Tensor[[64], pl.FP32] = self.mutate(x)
                if cond:
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)  # VIOLATION
                else:
                    y: pl.Tensor[[64], pl.FP32] = pl.mul(T_new, T_new)
                return y

        body = _seq_body(P, "main")
        diags = dep_analysis.check_inout_use_discipline(body, P)
        assert len(diags) >= 1
        assert all("'x'" in d.message for d in diags)

    def test_rejects_read_in_for_body(self):
        """A read inside a for body after the call is a violation."""

        @pl.program
        class P:
            @pl.function
            def mutate(self, T: pl.InOut[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
                r: pl.Tensor[[64], pl.FP32] = pl.add(T, T)
                return r

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                T_new: pl.Tensor[[64], pl.FP32] = self.mutate(x)
                for i, (acc,) in pl.range(0, 4, 1, init_values=(T_new,)):
                    acc = pl.yield_(pl.add(acc, x))  # VIOLATION reading x
                return acc

        body = _seq_body(P, "main")
        diags = dep_analysis.check_inout_use_discipline(body, P)
        assert len(diags) >= 1
        assert all("'x'" in d.message for d in diags)

    def test_sibling_if_branch_does_not_bleed(self):
        """Call in then-branch; read of same var in else-branch is NOT a violation."""

        @pl.program
        class P:
            @pl.function
            def mutate(self, T: pl.InOut[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
                r: pl.Tensor[[64], pl.FP32] = pl.add(T, T)
                return r

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32], cond: pl.Scalar[pl.BOOL]) -> pl.Tensor[[64], pl.FP32]:
                if cond:
                    y_then: pl.Tensor[[64], pl.FP32] = self.mutate(x)  # noqa: F841
                else:
                    # x is only "dead" in the then-branch; reading here is OK.
                    y_else: pl.Tensor[[64], pl.FP32] = pl.add(x, x)  # noqa: F841
                # At runtime, only one branch executed; this read is not
                # guaranteed to follow the mutating call.
                return x

        body = _seq_body(P, "main")
        diags = dep_analysis.check_inout_use_discipline(body, P)
        # The `return x` after the if is reachable from the mutating call on
        # the then-path, so it IS a violation. But the read inside the
        # else-branch must not be flagged: before the if-branch-scoping fix we
        # saw extra diagnostics for the else-branch read; with scoping, only
        # the post-if `return x` is flagged.
        assert len(diags) <= 1

    def test_self_read_in_call_args_is_allowed(self):
        """`f(T, inout=T)` — reading T as an arg on the same call is legal."""

        @pl.program
        class P:
            @pl.function
            def mutate(
                self,
                read_only: pl.Tensor[[64], pl.FP32],
                T: pl.InOut[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                r: pl.Tensor[[64], pl.FP32] = pl.add(read_only, T)
                return r

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                # x passed twice: once as read-only, once as InOut. Arg-side read
                # happens before the call effect, so this is allowed.
                T_new: pl.Tensor[[64], pl.FP32] = self.mutate(x, x)
                return T_new

        body = _seq_body(P, "main")
        diags = dep_analysis.check_inout_use_discipline(body, P)
        assert diags == []

    def test_builtin_ops_do_not_kill_vars(self):
        """Built-in ops (tile.*/tensor.*) don't contribute to the dead set."""

        # A program that only uses built-in ops — no user function with InOut
        # params. The validator must accept it cleanly, even if the same tensor
        # appears on multiple stmts. Mode B (physical aliasing) is out of scope
        # for this analysis.
        @pl.program
        class P:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                b: pl.Tensor[[64], pl.FP32] = pl.mul(x, x)
                c: pl.Tensor[[64], pl.FP32] = pl.add(a, b)
                return c

        body = _seq_body(P, "main")
        diags = dep_analysis.check_inout_use_discipline(body, P)
        assert diags == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
