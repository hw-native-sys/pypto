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

Tests use the @pl.program DSL where possible. Constant folding tests use
direct IR construction because Python eagerly evaluates constant expressions
(e.g., `3 + 4` becomes `7` before the DSL sees it).
"""

import pypto.language as pl
import pytest
from pypto import DataType, ir, passes

S = ir.Span.unknown()
IDX = DataType.INDEX


def ci(value: int, dtype: DataType = IDX) -> ir.ConstInt:
    return ir.ConstInt(value, dtype, S)


def make_var(name: str, dtype: DataType = IDX) -> ir.Var:
    return ir.Var(name, ir.ScalarType(dtype), S)


def wrap_stmts(stmts):
    if len(stmts) == 1:
        return stmts[0]
    return ir.SeqStmts(stmts, S)


def make_program(body_stmts, return_types=None):
    """Build a single-function Program.

    Pass `return_types` when `body_stmts` ends in a ReturnStmt so the
    function signature matches the returned values (otherwise the
    structural check complains about an empty signature).
    """
    body = wrap_stmts(body_stmts)
    func = ir.Function("main", [], return_types or [], body, S)
    return ir.Program([func], "test", S)


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
# Constant folding (direct IR — DSL can't produce constant-only expressions)
# ============================================================================


class TestConstantFolding:
    """Verify arithmetic constant folding — tests put the expression
    directly in a ReturnStmt so the fold result stays observable after
    Simplify's scalar DCE step."""

    def test_add_constants(self):
        """3 + 4 should fold to 7."""
        ret_type = ir.ScalarType(IDX)
        before = make_program([ir.ReturnStmt([ir.Add(ci(3), ci(4), IDX, S)], S)], [ret_type])
        expected = make_program([ir.ReturnStmt([ci(7)], S)], [ret_type])

        after = passes.simplify()(before)
        ir.assert_structural_equal(after, expected)

    def test_mul_constants(self):
        """3 * 4 should fold to 12."""
        ret_type = ir.ScalarType(IDX)
        before = make_program([ir.ReturnStmt([ir.Mul(ci(3), ci(4), IDX, S)], S)], [ret_type])
        expected = make_program([ir.ReturnStmt([ci(12)], S)], [ret_type])

        after = passes.simplify()(before)
        ir.assert_structural_equal(after, expected)

    def test_nested_constant_expr(self):
        """(2 + 3) * 4 should fold to 20."""
        ret_type = ir.ScalarType(IDX)
        inner = ir.Add(ci(2), ci(3), IDX, S)
        outer = ir.Mul(inner, ci(4), IDX, S)
        before = make_program([ir.ReturnStmt([outer], S)], [ret_type])
        expected = make_program([ir.ReturnStmt([ci(20)], S)], [ret_type])

        after = passes.simplify()(before)
        ir.assert_structural_equal(after, expected)


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
        body = ir.SeqStmts([], S)
        func = ir.Function("main", [], [], body, S)
        before = ir.Program([func], "test", S)

        after = passes.simplify()(before)
        ir.assert_structural_equal(after, before)


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
# Scalar dead-code elimination (conservative — preserves Call-RHS assigns)
# ============================================================================


class TestScalarDCE:
    """The final step of Simplify is a conservative scalar DCE. It removes
    AssignStmts whose LHS is scalar and whose RHS is not a Call, provided
    the LHS has no remaining uses. Call-backed and tensor-typed assigns
    are always preserved — the IR has no purity annotation yet."""

    def test_removes_unused_scalar_const(self):
        """A scalar constant with no uses is removed."""
        y = make_var("y")
        before = make_program([ir.AssignStmt(y, ci(5), S)])
        expected = make_program([])

        after = passes.simplify()(before)
        ir.assert_structural_equal(after, expected)

    def test_cascade_scalar_chain(self):
        """`a = 5; b = a + 1` with b unused removes both."""
        a = make_var("a")
        b = make_var("b")
        stmts = [
            ir.AssignStmt(a, ci(5), S),
            ir.AssignStmt(b, ir.Add(a, ci(1), IDX, S), S),
        ]
        before = make_program(stmts)
        expected = make_program([])

        after = passes.simplify()(before)
        ir.assert_structural_equal(after, expected)

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

        Uses a synthetic Op name that won't roundtrip through the DSL parser,
        so we run under a roundtrip-free PassContext to exercise the DCE
        predicate directly.
        """
        y = make_var("y", DataType.INT64)
        call = ir.Call(ir.Op("test.pure_scalar"), [], ir.ScalarType(DataType.INT64), S)
        before = make_program([ir.AssignStmt(y, call, S)])

        with passes.PassContext([]):
            after = passes.simplify()(before)
        # LHS is scalar-typed and unused, but the direct-Call RHS keeps it.
        ir.assert_structural_equal(after, before)

    def test_keeps_scalar_assign_with_nested_call_rhs(self):
        """A scalar LHS whose RHS contains a Call nested inside an arithmetic
        expression must be preserved — any expression containing a Call may
        have side effects, not just a top-level Call."""
        y = make_var("y", DataType.INT64)
        call = ir.Call(ir.Op("test.pure_scalar"), [], ir.ScalarType(DataType.INT64), S)
        nested = ir.Add(call, ci(1, DataType.INT64), DataType.INT64, S)
        before = make_program([ir.AssignStmt(y, nested, S)])

        with passes.PassContext([]):
            after = passes.simplify()(before)
        # Nested Call must still block removal.
        ir.assert_structural_equal(after, before)

    def test_drops_dead_scalar_inside_scope(self):
        """An unused scalar inside a ScopeStmt body is removed — DCE recurses
        into scope bodies, not just For/If/While.

        An evaluation statement anchors the scope so its body stays non-empty
        after DCE (an empty scope body is not representable in the DSL).
        """
        dead = make_var("dead")
        dead_assign = ir.AssignStmt(dead, ci(7), S)
        anchor = ir.EvalStmt(
            ir.Call(ir.Op("system.sync"), [], ir.ScalarType(IDX), S),
            S,
        )
        scope_before = ir.InCoreScopeStmt(body=ir.SeqStmts([dead_assign, anchor], S), span=S)
        before = make_program([scope_before])

        scope_expected = ir.InCoreScopeStmt(body=anchor, span=S)
        expected = make_program([scope_expected])

        # Synthetic Op names don't roundtrip; run without the roundtrip check.
        with passes.PassContext([]):
            after = passes.simplify()(before)
        ir.assert_structural_equal(after, expected)


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
