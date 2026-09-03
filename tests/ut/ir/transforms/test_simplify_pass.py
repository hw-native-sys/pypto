# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for the Simplify pass.

This pass simplifies expressions and statements in the IR using algebraic
rewrite rules and bound analysis. IRMutatorWithAnalyzer binds ForStmt loop
variables to their ranges, and ConstraintContext propagates if-branch
conditions, enabling range-aware simplification.

Tests use the @pl.program DSL. Constant-folding tests author un-folded
constant expressions with ``pl.const(value, dtype)`` — each call builds a
distinct ``ConstInt`` IR node, so ``pl.const(3, ...) + pl.const(4, ...)``
reaches the parser as an un-evaluated ``Add`` (Python never sees two bare
literals to pre-fold).
"""

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto import ir, passes

_OP_PLD_TENSOR_ALLREDUCE = ir.get_op("pld.tensor.allreduce").name

# ============================================================================
# Pass metadata
# ============================================================================


class TestPassMetadata:
    def test_pass_name(self):
        p = passes.simplify()
        assert p.get_name() == "Simplify"

    def test_pass_no_required_properties(self):
        p = passes.simplify()
        assert p.get_required_properties().empty()

    def test_pass_no_produced_properties(self):
        p = passes.simplify()
        assert p.get_produced_properties().empty()


# ============================================================================
# Identity simplifications (x + 0 -> x, x * 1 -> x)
# ============================================================================


class TestIdentitySimplification:
    """Scalars are written into a tensor sink so DCE does not prune them
    and the fold result stays observable in the IR."""

    def test_add_zero(self):
        """x + 0 should simplify to x."""

        @pl.program
        class Before:
            @pl.function
            def main(self, out: pl.Tensor[[8], pl.INT64]):
                for i in pl.range(8):
                    y: pl.Scalar[pl.INT64] = i + 0
                    pl.tensor.write(out, [i], y)

        @pl.program
        class Expected:
            @pl.function
            def main(self, out: pl.Tensor[[8], pl.INT64]):
                for i in pl.range(8):
                    y: pl.Scalar[pl.INT64] = i
                    pl.tensor.write(out, [i], y)

        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Expected)

    def test_zero_add(self):
        """0 + x should simplify to x."""

        @pl.program
        class Before:
            @pl.function
            def main(self, out: pl.Tensor[[8], pl.INT64]):
                for i in pl.range(8):
                    y: pl.Scalar[pl.INT64] = 0 + i
                    pl.tensor.write(out, [i], y)

        @pl.program
        class Expected:
            @pl.function
            def main(self, out: pl.Tensor[[8], pl.INT64]):
                for i in pl.range(8):
                    y: pl.Scalar[pl.INT64] = i
                    pl.tensor.write(out, [i], y)

        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Expected)

    def test_mul_one(self):
        """x * 1 should simplify to x."""

        @pl.program
        class Before:
            @pl.function
            def main(self, out: pl.Tensor[[8], pl.INT64]):
                for i in pl.range(8):
                    y: pl.Scalar[pl.INT64] = i * 1
                    pl.tensor.write(out, [i], y)

        @pl.program
        class Expected:
            @pl.function
            def main(self, out: pl.Tensor[[8], pl.INT64]):
                for i in pl.range(8):
                    y: pl.Scalar[pl.INT64] = i
                    pl.tensor.write(out, [i], y)

        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Expected)

    def test_sub_zero(self):
        """x - 0 should simplify to x."""

        @pl.program
        class Before:
            @pl.function
            def main(self, out: pl.Tensor[[8], pl.INT64]):
                for i in pl.range(8):
                    y: pl.Scalar[pl.INT64] = i - 0
                    pl.tensor.write(out, [i], y)

        @pl.program
        class Expected:
            @pl.function
            def main(self, out: pl.Tensor[[8], pl.INT64]):
                for i in pl.range(8):
                    y: pl.Scalar[pl.INT64] = i
                    pl.tensor.write(out, [i], y)

        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Expected)


# ============================================================================
# Constant folding
# ============================================================================


class TestConstantFolding:
    """Verify arithmetic constant folding — tests put the expression
    directly in a ReturnStmt so the fold result stays observable after
    Simplify's scalar DCE step.

    ``pl.const(value, dtype)`` builds a single ``ConstInt`` IR node, so an
    expression like ``pl.const(3, ...) + pl.const(4, ...)`` reaches the
    parser as an un-folded ``Add`` (Python never sees two bare literals to
    fold) — letting these stay style-A ``@pl.program`` tests.
    """

    def test_add_constants(self):
        """3 + 4 should fold to 7."""

        @pl.program
        class Before:
            @pl.function
            def main(self) -> pl.Scalar[pl.INDEX]:
                return pl.const(3, pl.INDEX) + pl.const(4, pl.INDEX)

        @pl.program
        class Expected:
            @pl.function
            def main(self) -> pl.Scalar[pl.INDEX]:
                return pl.const(7, pl.INDEX)

        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Expected)

    def test_mul_constants(self):
        """3 * 4 should fold to 12."""

        @pl.program
        class Before:
            @pl.function
            def main(self) -> pl.Scalar[pl.INDEX]:
                return pl.const(3, pl.INDEX) * pl.const(4, pl.INDEX)

        @pl.program
        class Expected:
            @pl.function
            def main(self) -> pl.Scalar[pl.INDEX]:
                return pl.const(12, pl.INDEX)

        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Expected)

    def test_nested_constant_expr(self):
        """(2 + 3) * 4 should fold to 20."""

        @pl.program
        class Before:
            @pl.function
            def main(self) -> pl.Scalar[pl.INDEX]:
                return (pl.const(2, pl.INDEX) + pl.const(3, pl.INDEX)) * pl.const(4, pl.INDEX)

        @pl.program
        class Expected:
            @pl.function
            def main(self) -> pl.Scalar[pl.INDEX]:
                return pl.const(20, pl.INDEX)

        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Expected)


# ============================================================================
# Range-aware simplification (requires loop variable binding)
# ============================================================================


class TestRangeAwareSimplification:
    def test_floordiv_by_range_bound(self):
        """i // 8 should simplify to 0 when i is in [0, 8)."""

        @pl.program
        class Before:
            @pl.function
            def main(self, out: pl.Tensor[[8], pl.INT64]):
                for i in pl.range(8):
                    y: pl.Scalar[pl.INT64] = i // 8
                    pl.tensor.write(out, [i], y)

        @pl.program
        class Expected:
            @pl.function
            def main(self, out: pl.Tensor[[8], pl.INT64]):
                for i in pl.range(8):
                    y: pl.Scalar[pl.INT64] = 0
                    pl.tensor.write(out, [i], y)

        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Expected)

    def test_floormod_by_range_bound(self):
        """i % 8 should simplify to i when i is in [0, 8)."""

        @pl.program
        class Before:
            @pl.function
            def main(self, out: pl.Tensor[[8], pl.INT64]):
                for i in pl.range(8):
                    y: pl.Scalar[pl.INT64] = i % 8
                    pl.tensor.write(out, [i], y)

        @pl.program
        class Expected:
            @pl.function
            def main(self, out: pl.Tensor[[8], pl.INT64]):
                for i in pl.range(8):
                    y: pl.Scalar[pl.INT64] = i
                    pl.tensor.write(out, [i], y)

        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Expected)

    def test_floordiv_not_simplifiable(self):
        """i // 4 should NOT simplify when i is in [0, 8) — result is 0 or 1."""

        @pl.program
        class Before:
            @pl.function
            def main(self, out: pl.Tensor[[8], pl.INT64]):
                for i in pl.range(8):
                    y: pl.Scalar[pl.INT64] = i // 4
                    pl.tensor.write(out, [i], y)

        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Before)

    def test_nested_loops(self):
        """Inner loop variable binding should work in nested loops."""

        @pl.program
        class Before:
            @pl.function
            def main(self, out: pl.Tensor[[8, 4], pl.INT64]):
                for i in pl.range(8):
                    for j in pl.range(4):
                        y: pl.Scalar[pl.INT64] = j // 4
                        pl.tensor.write(out, [i, j], y)

        @pl.program
        class Expected:
            @pl.function
            def main(self, out: pl.Tensor[[8, 4], pl.INT64]):
                for i in pl.range(8):
                    for j in pl.range(4):
                        y: pl.Scalar[pl.INT64] = 0
                        pl.tensor.write(out, [i, j], y)

        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Expected)


# ============================================================================
# If-branch constraint propagation
# ============================================================================


class TestIfBranchConstraint:
    def test_then_branch_uses_condition(self):
        """In then-branch of `if i < 4`, i is in [0, 4) so i // 4 == 0."""

        @pl.program
        class Before:
            @pl.function
            def main(self, out: pl.Tensor[[8], pl.INT64]):
                for i in pl.range(8):
                    if i < 4:
                        y: pl.Scalar[pl.INT64] = i // 4
                        pl.tensor.write(out, [i], y)

        @pl.program
        class Expected:
            @pl.function
            def main(self, out: pl.Tensor[[8], pl.INT64]):
                for i in pl.range(8):
                    if i < 4:
                        y: pl.Scalar[pl.INT64] = 0
                        pl.tensor.write(out, [i], y)

        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Expected)

    def test_else_branch_uses_negated_condition(self):
        """In else-branch of `if i < 4`, Not(i<4) → i>=4 tightens bounds to [4, 8).
        Combined with loop [0, 8): i // 8 ∈ [0, 0] → 0."""

        @pl.program
        class Before:
            @pl.function
            def main(self, out: pl.Tensor[[8], pl.INT64]):
                for i in pl.range(8):
                    if i < 4:
                        y: pl.Scalar[pl.INT64] = i // 4
                        pl.tensor.write(out, [i], y)
                    else:
                        y2: pl.Scalar[pl.INT64] = i // 8
                        pl.tensor.write(out, [i], y2)

        @pl.program
        class Expected:
            @pl.function
            def main(self, out: pl.Tensor[[8], pl.INT64]):
                for i in pl.range(8):
                    if i < 4:
                        y: pl.Scalar[pl.INT64] = 0
                        pl.tensor.write(out, [i], y)
                    else:
                        y2: pl.Scalar[pl.INT64] = 0
                        pl.tensor.write(out, [i], y2)

        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Expected)

    def test_nested_if_in_loop(self):
        """Nested if inside for loop: both loop binding and condition constraint active."""

        @pl.program
        class Before:
            @pl.function
            def main(self, out: pl.Tensor[[16], pl.INT64]):
                for i in pl.range(16):
                    if i < 8:
                        y: pl.Scalar[pl.INT64] = i // 8
                        pl.tensor.write(out, [i], y)
                    else:
                        z: pl.Scalar[pl.INT64] = i // 16
                        pl.tensor.write(out, [i], z)

        @pl.program
        class Expected:
            @pl.function
            def main(self, out: pl.Tensor[[16], pl.INT64]):
                for i in pl.range(16):
                    if i < 8:
                        y: pl.Scalar[pl.INT64] = 0
                        pl.tensor.write(out, [i], y)
                    else:
                        z: pl.Scalar[pl.INT64] = 0
                        pl.tensor.write(out, [i], z)

        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Expected)


# ============================================================================
# Comprehensive control flow (break, continue, scope, while, seq)
# ============================================================================


class TestControlFlow:
    def test_break_stmt_passthrough(self):
        """BreakStmt is a leaf — pass should simplify surrounding exprs without error."""

        @pl.program
        class Before:
            @pl.function
            def main(self, out: pl.Tensor[[8], pl.INT64]):
                for i in pl.range(8):
                    y: pl.Scalar[pl.INT64] = i + 0
                    pl.tensor.write(out, [i], y)
                    break

        @pl.program
        class Expected:
            @pl.function
            def main(self, out: pl.Tensor[[8], pl.INT64]):
                for i in pl.range(8):
                    y: pl.Scalar[pl.INT64] = i
                    pl.tensor.write(out, [i], y)
                    break

        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Expected)

    def test_continue_stmt_passthrough(self):
        """ContinueStmt is a leaf — pass should simplify surrounding exprs without error."""

        @pl.program
        class Before:
            @pl.function
            def main(self, out: pl.Tensor[[4], pl.INT64]):
                for i in pl.range(4):
                    y: pl.Scalar[pl.INT64] = i * 1
                    pl.tensor.write(out, [i], y)
                    continue

        @pl.program
        class Expected:
            @pl.function
            def main(self, out: pl.Tensor[[4], pl.INT64]):
                for i in pl.range(4):
                    y: pl.Scalar[pl.INT64] = i
                    pl.tensor.write(out, [i], y)
                    continue

        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Expected)

    def test_scope_stmt_traversal(self):
        """Pass should traverse into ScopeStmt bodies and simplify."""

        @pl.program
        class Before:
            @pl.function
            def main(self, out: pl.Tensor[[8], pl.INT64]):
                for i in pl.range(8):
                    with pl.at(level=pl.Level.CORE_GROUP):
                        y: pl.Scalar[pl.INT64] = i + 0
                        pl.tensor.write(out, [i], y)

        @pl.program
        class Expected:
            @pl.function
            def main(self, out: pl.Tensor[[8], pl.INT64]):
                for i in pl.range(8):
                    with pl.at(level=pl.Level.CORE_GROUP):
                        y: pl.Scalar[pl.INT64] = i
                        pl.tensor.write(out, [i], y)

        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Expected)

    def test_while_condition_simplified(self):
        """WhileStmt condition expressions should be simplified."""

        @pl.program
        class Before:
            @pl.function
            def main(self, n: pl.Scalar[pl.INDEX]):
                i: pl.Scalar[pl.INDEX] = 0
                while i < n + 0:
                    i = i + 1

        @pl.program
        class Expected:
            @pl.function
            def main(self, n: pl.Scalar[pl.INDEX]):
                i: pl.Scalar[pl.INDEX] = 0
                while i < n:
                    i = i + 1

        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Expected)

    def test_while_iter_arg_stays_one_node_when_its_init_folds(self):
        """A WhileStmt IterArg must not split into a header node + orphan uses.

        An ``IterArg`` *use* is the same node as its declaration and carries
        ``initValue_``, so ``IRMutator::VisitExpr_(IterArgPtr)`` mints a fresh
        IterArg at the first use whose init the analyzer rewrote. Here
        ``i = 0`` is a top-level constant, so ``VisitStmt_(AssignStmtPtr)``
        full-binds it and the init folds ``i -> 0``.

        ``ForStmt`` rebuilds ``iter_args_`` before its body so every reference
        resolves to the header's new node. ``WhileStmt`` did not: the header
        kept the stale IterArg while all four body/condition uses pointed at an
        undefined clone, which ``UseAfterDef`` reported four times.
        """

        @pl.program
        class Before:
            @pl.function
            def main(self, out: pl.Tensor[[8], pl.INDEX]):
                i: pl.Scalar[pl.INDEX] = 0
                for (i_it,) in pl.while_(init_values=(i,)):
                    pl.cond(i_it < 4)
                    y: pl.Scalar[pl.INDEX] = i_it * 2
                    pl.tensor.write(out, [i_it], y)
                    nxt: pl.Scalar[pl.INDEX] = i_it + 1
                    i_end: pl.Scalar[pl.INDEX] = pl.yield_(nxt)
                # Reading the return_var after the loop keeps the carry live and
                # exercises the return_vars_ rebuild alongside the iter_args_ one.
                pl.tensor.write(out, [7], i_end)

        after = passes.simplify()(Before)

        props = passes.IRPropertySet()
        props.insert(passes.IRProperty.UseAfterDef)
        diagnostics = passes.PropertyVerifierRegistry.verify(props, after)
        errors = [d for d in diagnostics if d.severity == passes.DiagnosticSeverity.Error]
        assert not errors, f"UseAfterDef errors after Simplify: {[d.message for d in errors]}"

        # `i = 0` folds into the iter_arg and is then DCE'd, so the body may be
        # the bare WhileStmt rather than a SeqStmts.
        func = next(iter(after.functions.values()))
        body = func.body
        stmts = body.stmts if isinstance(body, ir.SeqStmts) else [body]
        while_stmt = next(s for s in stmts if isinstance(s, ir.WhileStmt))
        iter_arg = while_stmt.iter_args[0]

        # The fold did happen — otherwise the test would pass vacuously.
        assert isinstance(iter_arg.initValue, ir.ConstInt)
        assert iter_arg.initValue.value == 0

        # ... and the condition reads that same node, not a clone of it.
        condition = while_stmt.condition
        assert isinstance(condition, ir.Lt)
        assert condition.left.same_as(iter_arg)

    def test_while_body_fold_does_not_leak_var_remap_past_the_loop(self):
        """A fold inside a while body must not rewrite uses after the loop.

        ``VisitScopedBody`` unbinds scalars but not ``var_remap_``. A nested fold
        records ``outer_var -> body-local value``: the single-trip inner
        ``pl.range(0, 1)`` fires Fold B, which binds ``acc_next -> acc + 1`` with
        ``acc`` substituted by its init ``i``.

        Leaking that past the loop rewrote the post-loop ``acc_next`` into
        ``i + 1`` — silently *wrong*, not merely dangling: ``acc_next`` holds what
        the last iteration computed, which equals the post-loop ``i``, so ``i + 1``
        is off by one, and ``i`` being in scope means no verifier flags it.

        The same program with a ``for`` as the outer loop is checked alongside it:
        ``ForStmt`` has snapshotted ``var_remap_`` around its body all along, so
        the two loop kinds must agree here.

        Pre-SSA on purpose — leak-mode bodies only exist before SSA conversion,
        and this pass runs at pipeline position 5 and again at 46.

        Verification stays on: ``LiftBodyToReturnVars`` materialises
        ``AssignStmt(acc_next, ...)`` inside the loop body for a return var whose
        uses outlive the enclosing ``var_remap_`` restore, so the post-loop
        reference keeps a definition. See
        ``test_fold_materializes_escaping_return_var``.
        """

        @pl.program
        class WhileOuter:
            @pl.function
            def main(self, out: pl.Tensor[[8], pl.INDEX]):
                i: pl.Scalar[pl.INDEX] = 0
                acc_next: pl.Scalar[pl.INDEX] = 0
                while i < 4:
                    for j, (acc,) in pl.range(0, 1, init_values=(i,)):
                        acc_next = pl.yield_(acc + 1)
                    i = i + 1
                pl.tensor.write(out, [0], acc_next)

        @pl.program
        class ForOuter:
            @pl.function
            def main(self, out: pl.Tensor[[8], pl.INDEX]):
                acc_next: pl.Scalar[pl.INDEX] = 0
                for k in pl.range(4):
                    for j, (acc,) in pl.range(0, 1, init_values=(k,)):
                        acc_next = pl.yield_(acc + 1)
                pl.tensor.write(out, [0], acc_next)

        def post_loop_operand(program):
            after = passes.simplify()(program)
            func = next(iter(after.functions.values()))
            body = func.body
            stmts = body.stmts if isinstance(body, ir.SeqStmts) else [body]
            call = next(s.expr for s in stmts if isinstance(s, ir.EvalStmt) and isinstance(s.expr, ir.Call))
            assert isinstance(call, ir.Call)
            return call.args[-1]

        for label, program in (("while", WhileOuter), ("for", ForOuter)):
            operand = post_loop_operand(program)
            # Before the fix the `while` case produced an `Add` here — the
            # loop-private `i + 1` substituted into a use outside the loop.
            assert isinstance(operand, ir.Var), (
                f"{label}: post-loop use was rewritten to {operand.as_python()}"
            )
            assert operand.name_hint.startswith("acc_next"), label

    def test_fold_materializes_escaping_return_var(self):
        """A folded loop's return var read after the loop gets a real definition.

        Fold B lifts a single-trip body by recording ``return_var -> yielded``
        in ``var_remap_``; the enclosing loop then rebases that map, so a
        leak-mode read *after* the loop would keep a Var the fold just stripped
        the only definition of — ``UseAfterDef``. ``ReturnVarEscapeIndex`` spots
        the escaping use, and the lift emits ``AssignStmt`` instead.

        The assignment must land inside the loop body: its RHS names the loop
        variable, so it cannot be hoisted past the loop — and the last iteration
        writing last is exactly the value a leak-mode post-loop read expects.
        """

        @pl.program
        class Before:
            @pl.function
            def main(self, out: pl.Tensor[[8], pl.INDEX]):
                acc_next: pl.Scalar[pl.INDEX] = 0
                for k in pl.range(4):
                    for j, (acc,) in pl.range(0, 1, init_values=(k,)):
                        acc_next = pl.yield_(acc + 1)
                pl.tensor.write(out, [0], acc_next)

        # Verification is on by default — the point of the test is that the
        # folded IR passes UseAfterDef.
        after = passes.simplify()(Before)
        func = next(iter(after.functions.values()))
        assert isinstance(func.body, ir.SeqStmts)
        stmts = func.body.stmts

        for_stmt = next(s for s in stmts if isinstance(s, ir.ForStmt))
        body = for_stmt.body
        body_stmts = body.stmts if isinstance(body, ir.SeqStmts) else [body]
        assign = next(s for s in body_stmts if isinstance(s, ir.AssignStmt))
        assert assign.var.name_hint.startswith("acc_next")
        # RHS is the yielded `acc + 1` with `acc` substituted by its init `k`.
        assert assign.value.as_python() == "k + 1", assign.value.as_python()

        # The post-loop read still names that same Var — not a substituted copy
        # of a loop-private expression.
        call = next(s.expr for s in stmts if isinstance(s, ir.EvalStmt) and isinstance(s.expr, ir.Call))
        assert isinstance(call.args[-1], ir.Var)
        assert call.args[-1].name_hint == assign.var.name_hint

    def test_sequential_stmts(self):
        """Multiple statements should all be simplified."""

        @pl.program
        class Before:
            @pl.function
            def main(self, out_y: pl.Tensor[[8], pl.INT64], out_z: pl.Tensor[[8], pl.INT64]):
                for i in pl.range(8):
                    y: pl.Scalar[pl.INT64] = i + 0
                    z: pl.Scalar[pl.INT64] = i * 1
                    pl.tensor.write(out_y, [i], y)
                    pl.tensor.write(out_z, [i], z)

        @pl.program
        class Expected:
            @pl.function
            def main(self, out_y: pl.Tensor[[8], pl.INT64], out_z: pl.Tensor[[8], pl.INT64]):
                for i in pl.range(8):
                    y: pl.Scalar[pl.INT64] = i
                    z: pl.Scalar[pl.INT64] = i
                    pl.tensor.write(out_y, [i], y)
                    pl.tensor.write(out_z, [i], z)

        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Expected)

    def test_if_with_break_and_continue(self):
        """If-branch with break/continue alongside simplifiable expressions."""

        @pl.program
        class Before:
            @pl.function
            def main(self, out: pl.Tensor[[8], pl.INT64]):
                for i in pl.range(8):
                    if i < 4:
                        y: pl.Scalar[pl.INT64] = i // 4
                        pl.tensor.write(out, [i], y)
                        break
                    else:
                        y2: pl.Scalar[pl.INT64] = i + 0
                        pl.tensor.write(out, [i], y2)
                        continue

        @pl.program
        class Expected:
            @pl.function
            def main(self, out: pl.Tensor[[8], pl.INT64]):
                for i in pl.range(8):
                    if i < 4:
                        y: pl.Scalar[pl.INT64] = 0
                        pl.tensor.write(out, [i], y)
                        break
                    else:
                        y2: pl.Scalar[pl.INT64] = i
                        pl.tensor.write(out, [i], y2)
                        continue

        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Expected)

    def test_for_loop_with_scope_and_if(self):
        """Complex nesting: for -> scope -> if with constraint propagation."""

        @pl.program
        class Before:
            @pl.function
            def main(self, out: pl.Tensor[[8], pl.INT64]):
                for i in pl.range(8):
                    with pl.at(level=pl.Level.CORE_GROUP):
                        if i < 4:
                            y: pl.Scalar[pl.INT64] = i // 4
                            pl.tensor.write(out, [i], y)

        @pl.program
        class Expected:
            @pl.function
            def main(self, out: pl.Tensor[[8], pl.INT64]):
                for i in pl.range(8):
                    with pl.at(level=pl.Level.CORE_GROUP):
                        if i < 4:
                            y: pl.Scalar[pl.INT64] = 0
                            pl.tensor.write(out, [i], y)

        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Expected)


# ============================================================================
# No-op cases
# ============================================================================


class TestNoChange:
    def test_already_simplified(self):
        """An already-simple expression should not change."""

        @pl.program
        class Before:
            @pl.function
            def main(self, out: pl.Tensor[[8], pl.INT64]):
                for i in pl.range(8):
                    y: pl.Scalar[pl.INT64] = i
                    pl.tensor.write(out, [i], y)

        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Before)

    def test_symbolic_loop_bounds(self):
        """Non-constant loop bounds: binding is skipped, identity simplification still works."""

        @pl.program
        class Before:
            @pl.function
            def main(self, n: pl.Scalar[pl.INDEX], out: pl.Tensor[[8], pl.INT64]):
                for i in pl.range(n):
                    y: pl.Scalar[pl.INT64] = i + 0
                    pl.tensor.write(out, [i], y)

        @pl.program
        class Expected:
            @pl.function
            def main(self, n: pl.Scalar[pl.INDEX], out: pl.Tensor[[8], pl.INT64]):
                for i in pl.range(n):
                    y: pl.Scalar[pl.INT64] = i
                    pl.tensor.write(out, [i], y)

        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Expected)

    def test_empty_function(self):
        """A function with no expressions should be unchanged."""

        @pl.program
        class Before:
            @pl.function
            def main(self):
                pass

        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Before)


# ============================================================================
# Scalar constant propagation
# ============================================================================


class TestScalarConstantPropagation:
    """Binding scalar assignments so downstream uses fold to the literal.

    Only safe for Vars assigned exactly once (SSA invariant), enforced by the
    MultiAssignCollector pre-pass so these tests work pre-SSA.
    """

    def test_propagates_into_subsequent_expr(self):
        """CHUNK_K = 512 should fold into CHUNK_K + 1 → 513."""

        @pl.program
        class Before:
            @pl.function
            def main(self, out: pl.Tensor[[1], pl.INDEX]):
                CHUNK_K: pl.Scalar[pl.INDEX] = 512
                y: pl.Scalar[pl.INDEX] = CHUNK_K + 1
                pl.tensor.write(out, [0], y)

        # After simplify + scalar DCE: 513 propagates into the write call,
        # and both CHUNK_K and y are dropped as dead scalar bindings.
        @pl.program
        class Expected:
            @pl.function
            def main(self, out: pl.Tensor[[1], pl.INDEX]):
                pl.tensor.write(out, [0], 513)  # pyright: ignore[reportArgumentType]

        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Expected)

    def test_propagates_into_for_bounds(self):
        """CHUNK_K bound to 512 should fold into pl.range(0, 1024, CHUNK_K)."""

        @pl.program
        class Before:
            @pl.function
            def main(self):
                CHUNK_K: pl.Scalar[pl.INDEX] = 512
                for _i in pl.range(0, 1024, CHUNK_K):
                    pass

        # After simplify + scalar DCE: 512 propagates into the for-step and
        # CHUNK_K becomes dead, so the binding is removed.
        @pl.program
        class Expected:
            @pl.function
            def main(self):
                for _i in pl.range(0, 1024, 512):
                    pass

        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Expected)

    def test_propagates_into_tensor_shape_annotation(self):
        """Var bound to 4 should fold into both the LHS type annotation and
        the RHS tensor-op call arguments."""

        @pl.program
        class Before:
            @pl.function
            def main(self):
                N: pl.Scalar[pl.INDEX] = 4
                _t: pl.Tensor[[N, 8], pl.FP32] = pl.tensor.create([N, 8], dtype=pl.FP32)

        # After simplify + scalar DCE: N folds into the tensor shape and
        # Call args, then its binding is dropped as dead scalar. `_t` is
        # Call-backed so its assignment is preserved despite being unused.
        @pl.program
        class Expected:
            @pl.function
            def main(self):
                _t: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.create([4, 8], dtype=pl.FP32)

        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Expected)

    def test_propagates_into_distributed_tensor_view_type(self):
        """Simplified view shapes refresh DistributedTensorType metadata."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pld.DistributedTensor[[4, 8], pl.FP32]):
                n: pl.Scalar[pl.INDEX] = 2
                _viewed = pl.tensor.view(x, [n * 2, 8])

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pld.DistributedTensor[[4, 8], pl.FP32]):
                _viewed = pl.tensor.view(x, [4, 8])

        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Expected)

    def test_folds_nested_arithmetic_in_call_args(self):
        """`K + 0` buried inside a tensor-op argument should fold to `K` even
        though Analyzer::Simplify does not recurse into Call/MakeTuple."""

        @pl.program
        class Before:
            @pl.function
            def main(self, k: pl.Scalar[pl.INDEX]):
                _t: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.create([1 * 1, k + 0 - k + 8], dtype=pl.FP32)

        @pl.program
        class Expected:
            @pl.function
            def main(self, k: pl.Scalar[pl.INDEX]):  # noqa: ARG002
                _t: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.create([1, 8], dtype=pl.FP32)

        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Expected)

    def test_not_propagated_when_assigned_in_branch(self):
        """A scalar assigned inside a conditional branch must NOT be bound —
        the assignment doesn't dominate uses outside the branch, so folding
        the literal would be incorrect on paths where the branch didn't run.
        """

        @pl.program
        class Before:
            @pl.function
            def main(self, cond: pl.Scalar[pl.BOOL], out: pl.Tensor[[1], pl.INDEX]):
                k: pl.Scalar[pl.INDEX] = 7
                if cond:
                    k = 5
                y: pl.Scalar[pl.INDEX] = k + 1
                pl.tensor.write(out, [0], y)

        # Expected: no folding of `k` — the binding inside the branch isn't
        # safe to propagate past the merge point. `k + 1` stays symbolic.
        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Before)

    def test_not_propagated_when_reassigned(self):
        """A Var reassigned inside the function must NOT be bound to its
        initial value — pre-SSA safety via MultiAssignCollector.
        """

        @pl.program
        class Before:
            @pl.function
            def main(self, n: pl.Scalar[pl.INDEX]):
                i: pl.Scalar[pl.INDEX] = 0
                while i < n:
                    i = i + 1

        # Expected: identical to Before (no folding of `i` to 0 because `i` is
        # reassigned inside the loop).
        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Before)

    def test_propagates_into_iter_arg_type(self):
        """Var bound to 4 should fold into a loop-carried iter_arg's type."""

        @pl.program
        class Before:
            @pl.function
            def main(self):
                N: pl.Scalar[pl.INDEX] = 4
                acc: pl.Tensor[[N, 8], pl.FP32] = pl.tensor.create([N, 8], dtype=pl.FP32)
                for _i, (acc_iter,) in pl.range(4, init_values=(acc,)):
                    acc_iter = pl.tensor.add(acc_iter, acc_iter)

        # After simplify + scalar DCE: N folds into every shape annotation
        # and Call arg, then its scalar binding is dropped. `acc` is
        # Call-backed so it survives despite being unused after the fold.
        @pl.program
        class Expected:
            @pl.function
            def main(self):
                acc: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.create([4, 8], dtype=pl.FP32)
                for _i, (acc_iter,) in pl.range(4, init_values=(acc,)):
                    acc_iter = pl.tensor.add(acc_iter, acc_iter)

        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Expected)


# ============================================================================
# Scalar dead-code elimination (conservative — preserves Call-RHS assigns)
# ============================================================================


class TestScalarDCE:
    """The final step of Simplify is a conservative scalar DCE. It removes
    AssignStmts whose LHS is scalar and whose RHS is not a Call, provided
    the LHS has no remaining uses. Call-backed and tensor-typed assigns
    are always preserved — the IR has no purity annotation yet."""

    def test_removes_unused_scalar_const(self):
        """A scalar constant with no uses is removed."""

        @pl.program
        class Before:
            @pl.function
            def main(self):
                y: pl.Scalar[pl.INDEX] = 5  # noqa: F841

        @pl.program
        class Expected:
            @pl.function
            def main(self):
                pass

        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Expected)

    def test_cascade_scalar_chain(self):
        """`a = 5; b = a + 1` with b unused removes both."""

        @pl.program
        class Before:
            @pl.function
            def main(self):
                a: pl.Scalar[pl.INDEX] = 5
                b: pl.Scalar[pl.INDEX] = a + 1  # noqa: F841

        @pl.program
        class Expected:
            @pl.function
            def main(self):
                pass

        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Expected)

    def test_keeps_call_rhs_even_if_lhs_unused(self):
        """A Call-backed assignment is preserved even when LHS is unused —
        the call might have side effects we cannot yet reason about."""

        @pl.program
        class Before:
            @pl.function
            def main(self):
                _t: pl.Tensor[[4], pl.FP32] = pl.tensor.create([4], dtype=pl.FP32)

        # _t is unused, but pl.tensor.create is a Call → preserved.
        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Before)

    def test_keeps_used_scalar(self):
        """A scalar referenced downstream is preserved even after the
        upstream binding's LHS gets constant-folded away."""

        @pl.program
        class Before:
            @pl.function
            def main(self, out: pl.Tensor[[8], pl.INDEX]):
                for i in pl.range(8):
                    y: pl.Scalar[pl.INDEX] = i + 1
                    pl.tensor.write(out, [i], y)

        after = passes.simplify()(Before)
        # y is referenced by the write — scalar DCE leaves it alone.
        ir.assert_structural_equal(after, Before)

    def test_keeps_scalar_assign_with_direct_call_rhs(self):
        """A scalar LHS whose RHS is a direct Call must be preserved even
        when the LHS has no further uses — the Call may have side effects.

        A cross-function call returning a scalar is a real ``ir.Call`` that
        the DSL expresses directly, so this stays a style-A ``@pl.program``
        test (no synthetic Op / roundtrip-free PassContext needed).
        """

        @pl.program
        class Before:
            @pl.function
            def helper(self) -> pl.Scalar[pl.INT64]:
                return pl.const(0, pl.INT64)

            @pl.function
            def main(self):
                y: pl.Scalar[pl.INT64] = self.helper()  # noqa: F841

        # y is scalar-typed and unused, but the direct-Call RHS keeps it.
        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Before)

    def test_keeps_scalar_assign_with_nested_call_rhs(self):
        """A scalar LHS whose RHS contains a Call nested inside an arithmetic
        expression must be preserved — any expression containing a Call may
        have side effects, not just a top-level Call."""

        @pl.program
        class Before:
            @pl.function
            def helper(self) -> pl.Scalar[pl.INT64]:
                return pl.const(0, pl.INT64)

            @pl.function
            def main(self):
                y: pl.Scalar[pl.INT64] = self.helper() + pl.const(1, pl.INT64)  # noqa: F841

        # Nested Call must still block removal.
        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Before)

    def test_drops_dead_scalar_inside_scope(self):
        """An unused scalar inside a ScopeStmt body is removed — DCE recurses
        into scope bodies, not just For/If/While.

        A Call-backed ``tensor.create`` anchors the scope so its body stays
        non-empty after DCE (an empty scope body is not representable in the
        DSL).
        """

        @pl.program
        class Before:
            @pl.function
            def main(self):
                with pl.at(level=pl.Level.CORE_GROUP):
                    dead: pl.Scalar[pl.INDEX] = 7  # noqa: F841
                    _t: pl.Tensor[[4], pl.FP32] = pl.tensor.create([4], dtype=pl.FP32)

        @pl.program
        class Expected:
            @pl.function
            def main(self):
                with pl.at(level=pl.Level.CORE_GROUP):
                    _t: pl.Tensor[[4], pl.FP32] = pl.tensor.create([4], dtype=pl.FP32)

        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Expected)


# ============================================================================
# Fold A: collapse IfStmt when the analyzer can prove the condition.
# ============================================================================


class TestConstantIfCollapse:
    def test_always_true_keeps_then_drops_else(self):
        """`if i < 100` with i ∈ [0, 8): analyzer proves true, then-body lifted."""

        @pl.program
        class Before:
            @pl.function
            def main(self, out: pl.Tensor[[8], pl.INDEX]):
                for i in pl.range(8):
                    if i < 100:
                        y: pl.Scalar[pl.INDEX] = i + 1
                        pl.tensor.write(out, [i], y)
                    else:
                        z: pl.Scalar[pl.INDEX] = 99
                        pl.tensor.write(out, [i], z)

        @pl.program
        class Expected:
            @pl.function
            def main(self, out: pl.Tensor[[8], pl.INDEX]):
                for i in pl.range(8):
                    y: pl.Scalar[pl.INDEX] = i + 1
                    pl.tensor.write(out, [i], y)

        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Expected)

    def test_always_false_keeps_else(self):
        """`if i == -1` with i ∈ [0, 8): analyzer proves false, else-body lifted.

        Mirrors the qwen3 paged-attention pattern where a chunked-loop guard
        becomes statically dead after constant propagation.
        """

        @pl.program
        class Before:
            @pl.function
            def main(self, out: pl.Tensor[[8], pl.INDEX]):
                for i in pl.range(0, 8, 2):
                    if i == -1:
                        y: pl.Scalar[pl.INDEX] = 99
                        pl.tensor.write(out, [i], y)
                    else:
                        y2: pl.Scalar[pl.INDEX] = i + 1
                        pl.tensor.write(out, [i], y2)

        @pl.program
        class Expected:
            @pl.function
            def main(self, out: pl.Tensor[[8], pl.INDEX]):
                for i in pl.range(0, 8, 2):
                    y2: pl.Scalar[pl.INDEX] = i + 1
                    pl.tensor.write(out, [i], y2)

        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Expected)

    def test_always_false_no_else_drops_if_entirely(self):
        """`if i == -1` with no else and i ∈ [0, 8): the whole IfStmt vanishes.

        Fold A's always-false / no-else / empty-return_vars edge case
        (simplify_pass.cpp:462-469): when the condition is provably false and
        there is no else branch, the kept branch is an empty body
        (``loop_repair::MakeBody({})``) — the IfStmt is dropped entirely rather
        than collapsed to a branch. The surrounding loop keeps only its other
        statement (here the trailing unconditional write), since an empty
        for-body is not representable in the DSL.
        """

        @pl.program
        class Before:
            @pl.function
            def main(self, out: pl.Tensor[[8], pl.INDEX]):
                for i in pl.range(0, 8, 2):
                    if i == -1:
                        y: pl.Scalar[pl.INDEX] = 99
                        pl.tensor.write(out, [i], y)
                    pl.tensor.write(out, [i], i)

        @pl.program
        class Expected:
            @pl.function
            def main(self, out: pl.Tensor[[8], pl.INDEX]):
                for i in pl.range(0, 8, 2):
                    pl.tensor.write(out, [i], i)

        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Expected)

    def test_always_false_via_loop_affine_scalar(self):
        """A dead `if` guarded by a scalar bound to a loop-affine expression folds.

        `off = i * 256 + 256` with i ∈ [0, 8) gives off ∈ [256, ...], so
        `off == 0` is statically false and the else branch is kept.

        Regression for the qwen3 down_proj chunk guard `if o0__ssa_v2_1 == 0`
        that survived Simplify: the pass only registered *constant* scalar
        bindings (so a symbolic affine RHS was never analyzed), and
        MultiAssignCollector flagged every loop-body assignment as unsafe.
        `off` is bound for ConstIntBound analysis only — not substituted — so
        a surviving use of it would still print as `off`.
        """

        @pl.program
        class Before:
            @pl.function
            def main(self, out: pl.Tensor[[8], pl.INDEX]):
                for i in pl.range(0, 8, 2):
                    off: pl.Scalar[pl.INDEX] = i * 256 + 256
                    if off == 0:
                        y: pl.Scalar[pl.INDEX] = 99
                        pl.tensor.write(out, [i], y)
                    else:
                        y2: pl.Scalar[pl.INDEX] = i + 1
                        pl.tensor.write(out, [i], y2)

        # `off` becomes dead once the always-false branch is dropped, so scalar
        # DCE removes it. `y2 = i + 1` is symbolic and inside the loop, so it is
        # bound for analysis only and kept as a scalar (not inlined).
        @pl.program
        class Expected:
            @pl.function
            def main(self, out: pl.Tensor[[8], pl.INDEX]):
                for i in pl.range(0, 8, 2):
                    y2: pl.Scalar[pl.INDEX] = i + 1
                    pl.tensor.write(out, [i], y2)

        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Expected)

    def test_leaf_index_parameter_keeps_its_guard(self):
        """A leaf INDEX *parameter* is not non-negative, so its guard survives.

        Nothing here proves `a >= 0`. `a` is a runtime scalar the caller
        supplies, `INDEX` is a signed type — codegen emits `arith.cmpi slt` for
        it — and being unassigned only means the analyzer never saw a value, not
        that the value is non-negative. Folding `if a < 0` away would silently
        change what the kernel computes for `a = -1`: the program writes `b`,
        the folded one writes `a`.

        Non-negativity has to come from somewhere real — an explicit binding, a
        loop's constant start, or the extent rule the shape proofs opt into —
        never from the dtype.
        """

        @pl.program
        class Before:
            @pl.function
            def main(self, a: pl.Scalar[pl.INDEX], b: pl.Scalar[pl.INDEX], out: pl.Tensor[[1], pl.INDEX]):
                if a < 0:
                    pl.tensor.write(out, [0], b)
                else:
                    pl.tensor.write(out, [0], a)

        after = passes.simplify()(Before)
        # Unchanged: both arms, and the guard between them, are still reachable.
        ir.assert_structural_equal(after, Before)

    def test_derived_index_scalar_keeps_negative_range_guard(self):
        """A *derived* INDEX scalar takes its range from its RHS, not the dtype default.

        `idx = a - b` is negative whenever `b > a`, so `if idx < 0` is a live
        guard. `BindScalarBound` must record the derived range instead of
        intersecting it with the INDEX default `[0, +inf)` — that intersection
        deletes a reachable negative range and folds the guard away as
        statically false (issue #2500).
        """

        @pl.program
        class Before:
            @pl.function
            def main(self, a: pl.Scalar[pl.INDEX], b: pl.Scalar[pl.INDEX], out: pl.Tensor[[1], pl.INDEX]):
                idx: pl.Scalar[pl.INDEX] = a - b
                if idx < 0:
                    pl.tensor.write(out, [0], a)
                else:
                    pl.tensor.write(out, [0], idx)

        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Before)

    def test_nested_guard_on_derived_index_scalar_is_preserved(self):
        """Nested `if pos >= 0: if pos < N:` keeps both guards (issue #2500).

        The outer guard is the only thing keeping a negative offset out of the
        inner body. Proving it statically true left the upper-bound check
        standing alone, which a negative `pos` passes — the read then went
        ahead against a clamped offset and the kernel silently returned wrong
        data. Both `IfStmt`s must survive.
        """

        @pl.program
        class Before:
            @pl.function
            def main(self, base: pl.Scalar[pl.INDEX], out: pl.Tensor[[1], pl.INDEX]):
                pos: pl.Scalar[pl.INDEX] = base - 1
                if pos >= 0:
                    if pos < 8:
                        pl.tensor.write(out, [0], pos)

        # Both guards survive; the outer one is only canonicalized `Ge` -> `Le`.
        @pl.program
        class Expected:
            @pl.function
            def main(self, base: pl.Scalar[pl.INDEX], out: pl.Tensor[[1], pl.INDEX]):
                pos: pl.Scalar[pl.INDEX] = base - 1
                if 0 <= pos:
                    if pos < 8:
                        pl.tensor.write(out, [0], pos)

        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Expected)

    def test_keeps_unprovable_condition(self):
        """`if i == 0` with i ∈ [0, 8): polarity unknown — IfStmt preserved."""

        @pl.program
        class Before:
            @pl.function
            def main(self, out: pl.Tensor[[8], pl.INDEX]):
                for i in pl.range(8):
                    if i == 0:
                        a: pl.Scalar[pl.INDEX] = i + 1
                        pl.tensor.write(out, [i], a)
                    else:
                        b: pl.Scalar[pl.INDEX] = i + 2
                        pl.tensor.write(out, [i], b)

        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Before)


# ============================================================================
# Fold B: collapse a pure ForStmt with provable trip count 0 or 1.
# ============================================================================


class TestSingleTripLoopCollapse:
    def test_single_iteration_lifts_body(self):
        """`for _i in pl.range(1)`: trip 1, body lifted to function level.

        Body holds a tensor.create so a Call-backed AssignStmt anchors the
        lifted body — DCE preserves Call assignments, which keeps the body's
        only stmt observable for structural equality.
        """

        @pl.program
        class Before:
            @pl.function
            def main(self):
                for _i in pl.range(1):
                    _t: pl.Tensor[[16], pl.FP32] = pl.tensor.create([16], dtype=pl.FP32)

        @pl.program
        class Expected:
            @pl.function
            def main(self):
                _t: pl.Tensor[[16], pl.FP32] = pl.tensor.create([16], dtype=pl.FP32)

        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Expected)

    def test_zero_trip_loop_drops_body_no_return_vars(self):
        """`for _i in pl.range(0, 0)`: trip 0 with no return vars collapses to
        an empty body.

        Fold B's zero-trip branch (simplify_pass.cpp:272-281) proves
        ``stop <= start`` for ``pl.range(0, 0)`` (step 1, so ``CanProveGreaterEqual
        (step, 1)`` holds), then emits one ``AssignStmt(return_vars[i], init)``
        per return var and drops the body. With an empty ``return_vars_`` the
        emitted vector is empty, so ``loop_repair::MakeBody({})`` yields an
        empty body and the whole loop vanishes — leaving the function body
        empty (the Call-backed ``tensor.create`` inside the dead loop is
        discarded with the body, since the body never executes).
        """

        @pl.program
        class Before:
            @pl.function
            def main(self):
                for _i in pl.range(0, 0):
                    _t: pl.Tensor[[16], pl.FP32] = pl.tensor.create([16], dtype=pl.FP32)

        @pl.program
        class Expected:
            @pl.function
            def main(self):
                pass

        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Expected)

    def test_keeps_multi_iteration_loop(self):
        """Trip > 1: ForStmt preserved (control test)."""

        @pl.program
        class Before:
            @pl.function
            def main(self):
                for _i in pl.range(4):
                    _t: pl.Tensor[[8], pl.FP32] = pl.tensor.create([8], dtype=pl.FP32)

        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Before)

    def test_keeps_parallel_loop_purity_guard(self):
        """Single-trip Parallel loop: purity guard refuses to collapse Parallel kind."""

        @pl.program
        class Before:
            @pl.function
            def main(self):
                for _i in pl.parallel(1):
                    _t: pl.Tensor[[8], pl.FP32] = pl.tensor.create([8], dtype=pl.FP32)

        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Before)


# ============================================================================
# Fold A composes with Fold B in a single Simplify run: Fold B substitutes
# loop_var with a literal, exposing always-true/always-false predicates that
# Fold A then collapses, all in one traversal.
# ============================================================================


class TestFoldComposition:
    def test_single_trip_loop_then_constant_if(self):
        """`for ko in pl.range(0, 128, 128): if ko == 0:` collapses fully.

        After Fold B substitutes ko → 0, the inner `0 == 0` reduces to
        ConstBool(true) and Fold A drops the IfStmt, leaving only the
        then-body's contents.
        """

        @pl.program
        class Before:
            @pl.function
            def main(self):
                for ko in pl.range(0, 128, 128):
                    if ko == 0:
                        _t: pl.Tensor[[16], pl.FP32] = pl.tensor.create([16], dtype=pl.FP32)
                    else:
                        _t2: pl.Tensor[[32], pl.FP32] = pl.tensor.create([32], dtype=pl.FP32)

        @pl.program
        class Expected:
            @pl.function
            def main(self):
                _t: pl.Tensor[[16], pl.FP32] = pl.tensor.create([16], dtype=pl.FP32)

        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Expected)

    def test_two_sibling_single_trip_loops_no_var_alias(self):
        """Regression: Fold B trips=1 must not leak body-internal var_remap_
        entries into sibling scope.

        ``MaybeRebuildVar`` and inner Fold A's ``LiftBodyToReturnVars`` write
        entries keyed by raw ``Var*`` of the cloned-body locals. After the
        Fold returns, those clones can be released (their AssignStmts were
        rebuilt or lifted), and ``make_shared<Var>`` in a subsequent sibling
        Fold B can recycle the same heap address — the stale remap then
        substitutes the new Var with an unrelated value, producing IR where
        an AssignStmt's LHS Var has the wrong type for its RHS.

        Mirrors the qwen3_decode q_proj pattern (two peeled K-loops at the
        same scope, each with two unrolled iterations gated by ``ko == 0`` /
        ``ko + 64 == 0``) where the second loop's ``tile.extract`` LHS got
        aliased onto the first loop's matmul Acc accumulator (see e67e1488
        regression).
        """

        @pl.program
        class Before:
            @pl.function
            def main(self):
                for ko in pl.range(0, 128, 128):
                    if ko == 0:
                        _t1: pl.Tensor[[16], pl.FP32] = pl.tensor.create([16], dtype=pl.FP32)
                    else:
                        _t2: pl.Tensor[[16], pl.FP32] = pl.tensor.create([16], dtype=pl.FP32)
                    if ko + 64 == 0:
                        _t3: pl.Tensor[[16], pl.FP32] = pl.tensor.create([16], dtype=pl.FP32)
                    else:
                        _t4: pl.Tensor[[16], pl.FP32] = pl.tensor.create([16], dtype=pl.FP32)
                for ko_1 in pl.range(0, 128, 128):
                    if ko_1 == 0:
                        _t5: pl.Tensor[[32], pl.FP32] = pl.tensor.create([32], dtype=pl.FP32)
                    else:
                        _t6: pl.Tensor[[32], pl.FP32] = pl.tensor.create([32], dtype=pl.FP32)
                    if ko_1 + 64 == 0:
                        _t7: pl.Tensor[[32], pl.FP32] = pl.tensor.create([32], dtype=pl.FP32)
                    else:
                        _t8: pl.Tensor[[32], pl.FP32] = pl.tensor.create([32], dtype=pl.FP32)

        @pl.program
        class Expected:
            @pl.function
            def main(self):
                _t1: pl.Tensor[[16], pl.FP32] = pl.tensor.create([16], dtype=pl.FP32)
                _t4: pl.Tensor[[16], pl.FP32] = pl.tensor.create([16], dtype=pl.FP32)
                _t5: pl.Tensor[[32], pl.FP32] = pl.tensor.create([32], dtype=pl.FP32)
                _t8: pl.Tensor[[32], pl.FP32] = pl.tensor.create([32], dtype=pl.FP32)

        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Expected)


# ============================================================================
# tensor.view folding (RFC #1300 P4-b)
# ============================================================================


class TestTensorViewFolding:
    """Simplify drops identity ``tensor.view`` reinterprets per RFC §3.3.

    ``pl.tensor.view`` is a thin DSL wrapper over the internal
    ``tensor.view`` IR op — a recognised attribute of the ``pl.tensor``
    namespace — and the op round-trips through print→parse, so these stay
    style-A (Before/Expected ``@pl.program``) tests.

    Layout encoding refresher (RFC §4.2): row-major ``[a, b]`` ND describes
    the same physical buffer as ``[b, a]`` DN-packed. The trailing-dim swap
    is the canonical pair the validity check accepts.

    Note on chain folding: folding ``view(view(x, ...), ...)`` →
    ``view(x, ...)`` is intentionally not implemented at this layer.
    After SSA the outer Call references its inner via a Var, not inline,
    so naive pointer inspection cannot see across the binding. A dedicated
    SSA-aware chain optimizer can be added if a real pipeline produces such
    chains.
    """

    def test_eliminates_identity_view(self):
        """``view(x, x.layout)`` simplifies to ``x``: target layout
        matches source layout, so the call is a no-op."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[8, 4], pl.FP32]) -> pl.Tensor[[8, 4], pl.FP32]:
                # x is bare ND [8, 4]; flipping to ND is identity.
                same: pl.Tensor[[8, 4], pl.FP32] = pl.tensor.view(x, layout=pl.TensorLayout.ND)
                return same

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[8, 4], pl.FP32]) -> pl.Tensor[[8, 4], pl.FP32]:
                # 21f11ecb dropped the alias-fold: the view Call still folds
                # to ``x``, but the ``same = x`` residual is no longer removed.
                same: pl.Tensor[[8, 4], pl.FP32, pl.TensorView(stride=[4, 1], layout=pl.TensorLayout.ND)] = x
                return same

        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Expected)

    def test_preserves_substantive_layout_flip(self):
        """Genuine ND → DN flip (with the auto trailing-pair swap) survives —
        Simplify only drops layout-tag identities."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self, x: pl.Tensor[[8, 4], pl.FP32]
            ) -> pl.Tensor[[4, 8], pl.FP32, pl.TensorView(stride=[1, 4], layout=pl.TensorLayout.DN)]:
                y: pl.Tensor[[4, 8], pl.FP32, pl.TensorView(stride=[1, 4], layout=pl.TensorLayout.DN)] = (
                    pl.tensor.view(x, layout=pl.TensorLayout.DN)
                )
                return y

        after = passes.simplify()(Before)
        # Substantive flip is not a layout-tag identity, so it is preserved.
        ir.assert_structural_equal(after, Before)


# ============================================================================
# SpmdScope core_num folding
# ============================================================================


class TestSpmdScopeCoreNum:
    """Simplify folds the ``core_num_`` expression of a pre-outline
    ``SpmdScopeStmt`` (simplify_pass.cpp:383-395, doc §Algorithm step 2 last
    bullet). Closure arithmetic such as ``MAX // TILE`` arrives as an un-folded
    ``FloorDiv`` after parsing; one Simplify pass reduces it to a literal so
    later outlining records a concrete ``core_num`` attr.

    ``pl.spmd(pl.const(8, ...) // pl.const(2, ...))`` reaches the parser as an
    un-folded ``FloorDiv`` (two distinct ``ConstInt`` nodes — Python never sees
    two bare literals to pre-fold), so this stays a style-A ``@pl.program``
    test. The with-form body is a single InCore kernel call, the historical
    ``SpmdScopeStmt(<call>)`` shape the parser preserves when no optimizations
    are passed.
    """

    def test_spmd_core_num_floordiv_folds(self):
        """`with pl.spmd(8 // 2)` → `with pl.spmd(4)`: only core_num_ changes."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                x_tile: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                out: pl.Tensor[[64], pl.FP32] = pl.store(x_tile, [0], out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.spmd(pl.const(8, pl.INDEX) // pl.const(2, pl.INDEX)):
                    out = self.kernel(x, out)
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                x_tile: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                out: pl.Tensor[[64], pl.FP32] = pl.store(x_tile, [0], out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.spmd(pl.const(4, pl.INDEX)):
                    out = self.kernel(x, out)
                return out

        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Expected)


# ============================================================================
# Submit-awareness: Simplify walks Submit args and deps (see
# .claude/rules/pass-submit-awareness.md). A pl.submit inside pl.manual_scope
# is a first-class Submit; folding must reach its args/types and its dep edges
# must keep TaskId scalars live.
# ============================================================================


class TestManualScopeSubmit:
    def test_folds_shape_into_submit_arg_preserving_submit(self):
        """A top-level constant folds into a tensor shape that feeds a
        pl.submit inside pl.manual_scope.

        ``N = 4`` propagates into ``pl.tensor.create([N, 8], ...)`` and into the
        ``Submit``'s positional-arg type (the base IRMutator walks Submit args,
        mutator.cpp:407-415, so the leaf folds reach the rebuilt arg type). The
        dead ``N`` scalar binding is then dropped by scalar DCE, while the
        Submit-backed assignment is preserved — Submit is call-like, so DCE
        never prunes it. The single-LHS ``res = pl.submit(...)`` form keeps the
        body to one statement (no trailing unused TaskId projection to DCE).
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(self, t: pl.Tensor[[4, 8], pl.FP32]) -> pl.Tensor[[4, 8], pl.FP32]:
                return t

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self):
                N: pl.Scalar[pl.INDEX] = 4
                t: pl.Tensor[[N, 8], pl.FP32] = pl.tensor.create([N, 8], dtype=pl.FP32)
                with pl.manual_scope():
                    res = pl.submit(self.kernel, t)  # noqa: F841

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(self, t: pl.Tensor[[4, 8], pl.FP32]) -> pl.Tensor[[4, 8], pl.FP32]:
                return t

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self):
                t: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.create([4, 8], dtype=pl.FP32)
                with pl.manual_scope():
                    res = pl.submit(self.kernel, t)  # noqa: F841

        after = passes.simplify()(Before)
        ir.assert_structural_equal(after, Expected)

    def test_submit_dep_keeps_taskid_scalar_alive(self):
        """A TaskId scalar referenced by a later Submit's ``deps_`` is NOT
        dropped by scalar DCE — Simplify walks ``Submit.deps_`` as part of the
        use-def chain (pass-submit-awareness.md rule 2; mutator.cpp:417-429).

        ``a_tid`` is bound from ``_submit_tmp[1]`` (a scalar TASK_ID, non-Call
        RHS — normally a DCE candidate) but is consumed by the second submit's
        ``deps=[a_tid]``. Because the dep edge is a real SSA use that the
        traversal sees, ``a_tid`` survives and the program is unchanged. If
        Simplify ignored ``deps_``, ``a_tid`` would look dead and be pruned.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(self, t: pl.Tensor[[4, 8], pl.FP32]) -> pl.Tensor[[4, 8], pl.FP32]:
                return t

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self):
                t: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.create([4, 8], dtype=pl.FP32)
                with pl.manual_scope():
                    a, a_tid = pl.submit(self.kernel, t)
                    res2 = pl.submit(self.kernel, a, deps=[a_tid])  # noqa: F841

        after = passes.simplify()(Before)
        # No foldable exprs; a_tid is kept alive solely by the second Submit's
        # deps_ edge, so the program is structurally unchanged.
        ir.assert_structural_equal(after, Before)


# ============================================================================
# Dead IfStmt phi return_vars — issue #1603
# After ConvertToSSA, an if/else rebinding the same name in both arms produces
# IfStmt::return_vars_ (a phi). When that phi has no downstream consumer the
# outlining pass captures it as a spurious return on the outlined function and
# orchestration codegen miscompiles. Simplify must DCE the dead phi.
# ============================================================================


class TestDeadIfReturnVarsDCE:
    def test_drops_empty_synthetic_else_with_dead_phi(self):
        """Pruning the only synthetic else yield also removes the else branch.

        ConvertToSSA creates an else that yields the pre-if value when a
        source-level if has no else.  Once the unused phi is pruned, keeping an
        engaged empty else would print as ``else: pass`` and reparse as no
        else, breaking structural round-trip verification.
        """

        @pl.program
        class Before:
            @pl.function
            def main(self, cond: pl.Scalar[pl.BOOL], out: pl.Tensor[[1], pl.INDEX]):
                value: pl.Scalar[pl.INDEX] = 0
                if cond:
                    value = 1
                    pl.tensor.write(out, [0], value)

        ssa_form = passes.convert_to_ssa()(Before)
        ssa_func = next(iter(ssa_form.functions.values()))
        ssa_if_stmts = [s for s in ir.flatten_to_stmts(ssa_func.body) if isinstance(s, ir.IfStmt)]
        assert len(ssa_if_stmts) == 1
        ssa_if = ssa_if_stmts[0]
        assert len(ssa_if.return_vars) == 1
        assert ssa_if.else_body is not None

        after = passes.simplify()(ssa_form)
        func_after = next(iter(after.functions.values()))
        if_stmts = [s for s in ir.flatten_to_stmts(func_after.body) if isinstance(s, ir.IfStmt)]
        assert len(if_stmts) == 1
        assert len(if_stmts[0].return_vars) == 0
        assert if_stmts[0].else_body is None

    def test_drops_dead_scalar_phi_from_unused_if_else_rebind(self):
        """Issue #1603 minimal repro: a Scalar[INDEX] rebound in both arms of
        an if/else with no downstream use. After convert_to_ssa() + simplify()
        the IfStmt carries no phi return_vars, and the dead branch-body
        scalar assigns are gone — but the side-effecting in-branch writes
        (which actually use the per-branch SSA names) survive.
        """

        @pl.program
        class Before:
            @pl.function
            def main(self, cond: pl.Scalar[pl.BOOL], out: pl.Tensor[[1], pl.INDEX]):
                if cond:
                    t1: pl.Scalar[pl.INDEX] = 1
                    pl.tensor.write(out, [0], t1)
                else:
                    t1: pl.Scalar[pl.INDEX] = 2
                    pl.tensor.write(out, [0], t1)

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, cond: pl.Scalar[pl.BOOL], out: pl.Tensor[[1], pl.INDEX]):
                if cond:
                    t1_0: pl.Scalar[pl.INDEX] = 1
                    pl.tensor.write(out, [0], t1_0)
                else:
                    t1_1: pl.Scalar[pl.INDEX] = 2
                    pl.tensor.write(out, [0], t1_1)

        ssa_form = passes.convert_to_ssa()(Before)
        after = passes.simplify()(ssa_form)
        ir.assert_structural_equal(after, Expected)

    def test_keeps_scalar_phi_with_downstream_use(self):
        """Same if/else rebinding shape, but with a downstream consumer of t1
        after the if/else. The phi must survive because the consumer reads it.
        """

        @pl.program
        class Before:
            @pl.function
            def main(self, cond: pl.Scalar[pl.BOOL], out: pl.Tensor[[1], pl.INDEX]):
                if cond:
                    t1: pl.Scalar[pl.INDEX] = 1
                else:
                    t1: pl.Scalar[pl.INDEX] = 2
                pl.tensor.write(out, [0], t1)

        ssa_form = passes.convert_to_ssa()(Before)
        after = passes.simplify()(ssa_form)
        func_after = next(iter(after.functions.values()))
        if_stmts = [s for s in ir.flatten_to_stmts(func_after.body) if isinstance(s, ir.IfStmt)]
        assert len(if_stmts) == 1
        assert len(if_stmts[0].return_vars) == 1, (
            "phi return_var must survive when t1 has a downstream user; "
            f"got return_vars={if_stmts[0].return_vars}"
        )

    def test_drops_dead_tensor_phi_keeping_side_effect_ops(self):
        """Tensor-typed dead phi: both branches do a tensor.write (side-effect
        op preserved by DCE) and rebind t. With no downstream use of t, the
        phi return_var is dropped, but the in-branch writes survive.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                cond: pl.Scalar[pl.BOOL],
                a: pl.Scalar[pl.INDEX],
                b: pl.Scalar[pl.INDEX],
                out: pl.Tensor[[1], pl.INDEX],
            ):
                if cond:
                    t: pl.Tensor[[1], pl.INDEX] = pl.tensor.create([1], dtype=pl.INDEX)
                    pl.tensor.write(out, [0], a)
                    pl.tensor.write(t, [0], a)
                else:
                    t: pl.Tensor[[1], pl.INDEX] = pl.tensor.create([1], dtype=pl.INDEX)
                    pl.tensor.write(out, [0], b)
                    pl.tensor.write(t, [0], b)

        ssa_form = passes.convert_to_ssa()(Before)
        after = passes.simplify()(ssa_form)
        func_after = next(iter(after.functions.values()))
        if_stmts = [s for s in ir.flatten_to_stmts(func_after.body) if isinstance(s, ir.IfStmt)]
        assert len(if_stmts) == 1
        assert len(if_stmts[0].return_vars) == 0, (
            "Tensor phi return_var must be dropped when t has no downstream user; "
            f"got return_vars={if_stmts[0].return_vars}"
        )
        # Side-effecting tensor.write to `out` must be preserved in both arms.
        then_stmts = ir.flatten_to_stmts(if_stmts[0].then_body)
        else_body = if_stmts[0].else_body
        assert else_body is not None
        else_stmts = ir.flatten_to_stmts(else_body)

        def has_tensor_write(stmts):
            for s in stmts:
                expr = getattr(s, "expr", None) or getattr(s, "value", None)
                op = getattr(expr, "op", None) if expr is not None else None
                if op is not None and getattr(op, "name", "") == "tensor.write":
                    return True
            return False

        assert has_tensor_write(then_stmts), "tensor.write side-effect in then branch must survive phi-prune"
        assert has_tensor_write(else_stmts), "tensor.write side-effect in else branch must survive phi-prune"


class TestDeadLoopCarryDCE:
    """Loop-carried slots (``iter_args_[i]`` / ``return_vars_[i]``) with no
    reader on either end are dropped, together with the matching yield slot.

    This is the loop half of the dead-phi rule above. Reusing one Python local
    across two scopes is what produces the shape: SSA seeds the second loop
    with the first scope's value, the body overwrites it on every trip, and
    nobody reads either end. Left in place, the carry makes the *earlier*
    scope's value live-out, which for a device scope forces a Scalar into the
    outlined kernel's return set — a shape the runtime cannot carry at all.
    """

    @staticmethod
    def _only_for_stmt(program):
        func = next(iter(program.functions.values()))
        for_stmts = [s for s in ir.flatten_to_stmts(func.body) if isinstance(s, ir.ForStmt)]
        assert len(for_stmts) == 1
        return for_stmts[0]

    def test_drops_dead_carry_from_reused_scalar_name(self):
        """``t`` is bound before the loop and rebound in the body before any
        read, with no post-loop use: the carry is dead on both ends and the
        ForStmt keeps no iter_arg for it.
        """

        @pl.program
        class Before:
            @pl.function
            def main(self, n: pl.Scalar[pl.INDEX], out: pl.Tensor[[8], pl.INDEX]):
                t: pl.Scalar[pl.INDEX] = n * 2
                pl.tensor.write(out, [0], t)
                for i in pl.range(4):
                    t: pl.Scalar[pl.INDEX] = i * 2
                    pl.tensor.write(out, [1], t)

        after = passes.simplify()(passes.convert_to_ssa()(Before))
        for_stmt = self._only_for_stmt(after)
        assert len(for_stmt.iter_args) == 0, (
            "a carry the body overwrites before reading, with no post-loop "
            f"reader, must be dropped; got iter_args={for_stmt.iter_args}"
        )
        assert len(for_stmt.return_vars) == 0, (
            f"return_vars must be dropped in lockstep; got {for_stmt.return_vars}"
        )

    def test_keeps_carry_read_in_body(self):
        """An accumulator reads the incoming value each trip — the carry is
        live inside the body even though nothing reads it after the loop.
        """

        @pl.program
        class Before:
            @pl.function
            def main(self, n: pl.Scalar[pl.INDEX], out: pl.Tensor[[8], pl.INDEX]):
                acc: pl.Scalar[pl.INDEX] = n
                for i in pl.range(4):
                    acc: pl.Scalar[pl.INDEX] = acc + i
                    pl.tensor.write(out, [1], acc)

        after = passes.simplify()(passes.convert_to_ssa()(Before))
        for_stmt = self._only_for_stmt(after)
        assert len(for_stmt.iter_args) == 1, (
            f"an accumulator carry must survive; got iter_args={for_stmt.iter_args}"
        )

    def test_keeps_carry_used_after_loop(self):
        """The body rebinds without reading, but the final value is read after
        the loop — the ``return_var`` end keeps the slot alive.
        """

        @pl.program
        class Before:
            @pl.function
            def main(self, n: pl.Scalar[pl.INDEX], out: pl.Tensor[[8], pl.INDEX]):
                t: pl.Scalar[pl.INDEX] = n * 2
                for i in pl.range(4):
                    t: pl.Scalar[pl.INDEX] = i * 2
                pl.tensor.write(out, [0], t)

        after = passes.simplify()(passes.convert_to_ssa()(Before))
        for_stmt = self._only_for_stmt(after)
        assert len(for_stmt.iter_args) == 1, (
            f"a carry read after the loop must survive; got iter_args={for_stmt.iter_args}"
        )
        assert len(for_stmt.return_vars) == 1, (
            f"return_vars must survive in lockstep; got {for_stmt.return_vars}"
        )


class TestDistributedWindowBufferRemap:
    def test_window_buffer_remapped_in_lockstep_with_scope_slot(self):
        """Simplify folding a synthesized signal's window-buffer size
        (``world_size * 1 * 4`` → ``world_size * 4``) must remap the
        ``DistributedTensorType.window_buffer_`` back-reference in lockstep
        with the ``CommDomainScopeStmt`` slot — both must point at the SAME
        post-fold ``WindowBuffer``. Regression: the type rebuild left
        ``window_buffer_`` on the pre-fold object, so
        ``DistributedCodegen::ScopeForWindowBuffer``'s pointer-identity scan
        failed with "not a slot of any open CommDomainScopeStmt".
        """

        @pl.program
        class P:
            @pl.function(type=pl.FunctionType.Orchestration)
            def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
                return data

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(self):
                data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
                data = pld.window(data_buf, [256], dtype=pl.FP32)
                for r in pl.range(pld.world_size()):
                    self.chip_orch(data, device=r)
                data = pld.tensor.allreduce(data, op=pld.ReduceOp.Sum)
                return data

        # NOTE: run under a BEFORE_AND_AFTER-only context (no print/parse
        # roundtrip) — the materialize pass's output wraps the body in
        # CommDomainScopeStmt and stamps window_buffer_ back-references on
        # view Vars, neither of which the printer/parser pair roundtrips
        # (same override the materialize test file applies to all its passes).
        # The in-memory structural check below is the point of this test.
        instruments: list[passes.PassInstrument] = [
            passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)
        ]
        with passes.PassContext(instruments):
            materialized = passes.materialize_comm_domain_scopes()(passes.synthesize_allreduce_signals()(P))
            simplified = passes.simplify()(materialized)
        host = next(f for f in simplified.functions.values() if f.name == "host_orch")

        # `ir.flatten_to_stmts` does not descend into CommDomainScopeStmt
        # bodies, so walk recursively (the materialize test file uses the same
        # shape for the same reason).
        def walk(stmt):
            out = [stmt]
            if isinstance(stmt, ir.SeqStmts):
                for child in stmt.stmts:
                    out.extend(walk(child))
            if isinstance(stmt, ir.ScopeStmt):
                out.extend(walk(stmt.body))
            if isinstance(stmt, ir.ForStmt):
                out.extend(walk(stmt.body))
            if isinstance(stmt, ir.WhileStmt):
                out.extend(walk(stmt.body))
            return out

        stmts = walk(host.body)

        scopes = [s for s in stmts if isinstance(s, ir.CommDomainScopeStmt)]
        assert len(scopes) == 1
        signal_slots = [
            slot for slot in scopes[0].slots if slot.base.name_hint.startswith("__allreduce_signal_buf_")
        ]
        assert len(signal_slots) == 1

        allreduce_assigns = [
            s
            for s in stmts
            if isinstance(s, ir.AssignStmt)
            and isinstance(s.value, ir.Call)
            and s.value.op.name == _OP_PLD_TENSOR_ALLREDUCE
        ]
        assert len(allreduce_assigns) == 1
        allreduce_call = allreduce_assigns[0].value
        assert isinstance(allreduce_call, ir.Call)
        signal_var = allreduce_call.args[1]
        assert isinstance(signal_var, ir.Var)
        signal_type = signal_var.type
        assert isinstance(signal_type, ir.DistributedTensorType)

        # The view Var's window_buffer back-reference must be the SAME object
        # as the scope slot (both post-fold).
        assert signal_type.window_buffer is signal_slots[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
