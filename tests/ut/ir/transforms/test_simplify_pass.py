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
import pytest
from pypto import ir, passes

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
# DistributedTensorType metadata
# ============================================================================


def test_simplify_distributed_tensor_metadata_preserves_dtt_only_fields():
    """DTT metadata expressions simplify without degrading the type to TensorType."""
    span = ir.Span.unknown()

    def const(value: int) -> ir.ConstInt:
        return ir.ConstInt(value, ir.DataType.INDEX, span)

    def plus_zero(value: int) -> ir.Expr:
        return ir.Add(const(value), const(0), ir.DataType.INDEX, span)

    def const_values(exprs: list[ir.Expr]) -> list[int]:
        assert all(isinstance(expr, ir.ConstInt) for expr in exprs)
        return [expr.value for expr in exprs if isinstance(expr, ir.ConstInt)]

    mem_base = ir.Var("mem", ir.PtrType(), span)
    memref = ir.MemRef(mem_base, 0, 128, span)
    tensor_view = ir.TensorView(
        stride=[plus_zero(1), plus_zero(4)],
        layout=ir.TensorLayout.DN,
        valid_shape=[plus_zero(3), plus_zero(8)],
        pad=ir.PadValue.min,
    )
    viewed_type = ir.DistributedTensorType(
        [plus_zero(4), plus_zero(8)], ir.DataType.FP32, memref, tensor_view
    )

    window_base = ir.Var("window", ir.PtrType(), span)
    window = ir.WindowBuffer(window_base, const(256), span=span)
    window_type = ir.DistributedTensorType([plus_zero(16)], ir.DataType.FP16, window)

    viewed_call = ir.Call(ir.Op("test.viewed_dtt"), [], viewed_type, span)
    window_call = ir.Call(ir.Op("test.window_dtt"), [], window_type, span)
    body = ir.SeqStmts([ir.EvalStmt(viewed_call, span), ir.EvalStmt(window_call, span)], span)
    before = ir.Program([ir.Function("main", [], [], body, span)], "SimplifyDTT", span)

    # A concrete WindowBuffer is deliberately not representable in source type
    # annotations. Keep structural pass verification, but omit the printer
    # roundtrip instrument for this post-materialization IR state.
    with passes.PassContext([], passes.VerificationLevel.BASIC):
        after = passes.simplify()(before)

    func_after = after.get_function("main")
    assert func_after is not None
    assert isinstance(func_after.body, ir.SeqStmts)
    viewed_stmt = func_after.body.stmts[0]
    window_stmt = func_after.body.stmts[1]
    assert isinstance(viewed_stmt, ir.EvalStmt)
    assert isinstance(window_stmt, ir.EvalStmt)
    viewed_after = viewed_stmt.expr.type
    window_after = window_stmt.expr.type

    assert isinstance(viewed_after, ir.DistributedTensorType)
    assert const_values(list(viewed_after.shape)) == [4, 8]
    assert viewed_after.memref is memref
    assert viewed_after.window_buffer is None
    assert viewed_after.tensor_view is not None
    assert const_values(list(viewed_after.tensor_view.valid_shape)) == [3, 8]
    assert const_values(list(viewed_after.tensor_view.stride)) == [1, 4]
    assert viewed_after.tensor_view.layout == ir.TensorLayout.DN
    assert viewed_after.tensor_view.pad == ir.PadValue.min

    assert isinstance(window_after, ir.DistributedTensorType)
    assert const_values(list(window_after.shape)) == [16]
    assert window_after.window_buffer is window


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

    def test_symbolic_index_scalar_keeps_nonneg_default_bound(self):
        """A symbolic INDEX scalar must keep its non-negative default bound.

        `idx = a - b` (a, b INDEX) has an unknown [-inf, +inf] range, but
        `idx` is INDEX-typed and therefore non-negative. BindScalarBound must
        intersect the RHS range with the dtype default rather than overwrite
        it — otherwise the uninformative RHS range erases the non-negativity
        and `if idx < 0` (statically false for an INDEX scalar) stops folding.
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

        # `idx < 0` is statically false (INDEX ≥ 0), so the then branch drops.
        # `idx` is bound for analysis only, not substituted, so the surviving
        # write still references it.
        @pl.program
        class Expected:
            @pl.function
            def main(self, a: pl.Scalar[pl.INDEX], b: pl.Scalar[pl.INDEX], out: pl.Tensor[[1], pl.INDEX]):  # noqa: ARG002
                idx: pl.Scalar[pl.INDEX] = a - b
                pl.tensor.write(out, [0], idx)

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
# tensor.as_layout folding (RFC #1300 P4-b)
# ============================================================================


class TestAsLayoutFolding:
    """Simplify drops identity ``tensor.as_layout`` reinterprets per RFC §3.3.

    ``pl.tensor.as_layout`` is a thin DSL wrapper over the internal
    ``tensor.as_layout`` IR op — a recognised attribute of the ``pl.tensor``
    namespace — and the op round-trips through print→parse, so these stay
    style-A (Before/Expected ``@pl.program``) tests.

    Layout encoding refresher (RFC §4.2): row-major ``[a, b]`` ND describes
    the same physical buffer as ``[b, a]`` DN-packed. The trailing-dim swap
    is the canonical pair the validity check accepts.

    Note on chain folding: folding ``as_layout(as_layout(x, ...), ...)`` →
    ``as_layout(x, ...)`` is intentionally not implemented at this layer.
    After SSA the outer Call references its inner via a Var, not inline,
    so naive pointer inspection cannot see across the binding. A dedicated
    SSA-aware chain optimizer can be added if a real pipeline produces such
    chains.
    """

    def test_eliminates_identity_as_layout(self):
        """``as_layout(x, x.layout)`` simplifies to ``x``: target layout
        matches source layout, so the call is a no-op."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[8, 4], pl.FP32]) -> pl.Tensor[[8, 4], pl.FP32]:
                # x is bare ND [8, 4]; flipping to ND is identity.
                same: pl.Tensor[[8, 4], pl.FP32] = pl.tensor.as_layout(x, layout=pl.TensorLayout.ND)
                return same

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[8, 4], pl.FP32]) -> pl.Tensor[[8, 4], pl.FP32]:
                # 21f11ecb dropped the alias-fold: the as_layout Call still folds
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
                    pl.tensor.as_layout(x, layout=pl.TensorLayout.DN)
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
