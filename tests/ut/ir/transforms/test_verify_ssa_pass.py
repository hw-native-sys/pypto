# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for SSA verification via run_verifier()."""

import pypto.language as pl
import pytest
from pypto import DataType, ir, passes
from pypto.ir import builder


def test_verify_ssa_valid():
    """Test SSA verification with valid SSA IR."""
    ib = builder.IRBuilder()

    with ib.function("test_valid_ssa") as f:
        a = f.param("a", ir.ScalarType(DataType.INT64))
        b = f.param("b", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))

        x = ib.let("x", a)
        _y = ib.let("y", b)
        z = ib.let("z", x)

        ib.return_stmt(z)

    func = f.get_result()
    program = ir.Program([func], "test_program", ir.Span.unknown())

    # Run verification using run_verifier
    verify_pass = passes.run_verifier()
    result_program = verify_pass(program)

    assert result_program is not None


def test_verify_ssa_multiple_assignment():
    """Test SSA verification detects multiple assignments."""
    ib = builder.IRBuilder()

    with ib.function("test_multiple_assignment") as f:
        a = f.param("a", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))

        _x = ib.let("x", a)
        x2 = ib.let("x", a)  # Second assignment violates SSA

        ib.return_stmt(x2)

    func = f.get_result()
    program = ir.Program([func], "test_program", ir.Span.unknown())

    verify_pass = passes.run_verifier()
    result_program = verify_pass(program)
    assert result_program is not None


def test_verify_ssa_name_shadowing():
    """Test SSA verification detects name shadowing."""
    ib = builder.IRBuilder()

    with ib.function("test_shadow") as f:
        a = f.param("a", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))

        outer_i = ib.let("i", a)

        loop_var = ib.var("i", ir.ScalarType(DataType.INDEX))  # Shadows outer 'i'
        with ib.for_loop(loop_var, 0, 5, 1):
            _tmp = ib.let("tmp", loop_var)

        ib.return_stmt(outer_i)

    func = f.get_result()
    program = ir.Program([func], "test_program", ir.Span.unknown())

    verify_pass = passes.run_verifier()
    result_program = verify_pass(program)
    assert result_program is not None


def test_verify_ssa_missing_yield():
    """Test SSA verification detects missing yield in ForStmt."""
    span = ir.Span.unknown()

    a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
    params: list[ir.Var] = [a]
    return_types: list[ir.Type] = [ir.ScalarType(DataType.INT64)]

    loop_var = ir.Var("i", ir.ScalarType(DataType.INDEX), span)
    iter_arg = ir.IterArg("sum", ir.ScalarType(DataType.INT64), a, span)
    body = ir.AssignStmt(ir.Var("dummy", ir.ScalarType(DataType.INT64), span), loop_var, span)  # No yield!
    result_var = ir.Var("result", ir.ScalarType(DataType.INT64), span)

    for_stmt = ir.ForStmt(
        loop_var,
        ir.ConstInt(0, DataType.INDEX, span),
        ir.ConstInt(10, DataType.INDEX, span),
        ir.ConstInt(1, DataType.INDEX, span),
        [iter_arg],
        body,
        [result_var],
        span,
    )

    func_body = ir.SeqStmts([for_stmt, ir.ReturnStmt([result_var], span)], span)
    func = ir.Function("test_missing_yield", params, return_types, func_body, span)
    program = ir.Program([func], "test_program", span)

    verify_pass = passes.run_verifier()
    with pytest.raises(Exception, match="must have YieldStmt"):
        verify_pass(program)


def test_verify_ssa_missing_else():
    """Test SSA verification detects missing else branch."""
    span = ir.Span.unknown()

    a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
    params: list[ir.Var] = [a]
    return_types: list[ir.Type] = [ir.ScalarType(DataType.INT64)]

    condition = ir.Gt(a, ir.ConstInt(0, DataType.INT64, span), DataType.BOOL, span)
    then_body = ir.YieldStmt([a], span)
    result_var = ir.Var("result", ir.ScalarType(DataType.INT64), span)

    if_stmt = ir.IfStmt(condition, then_body, None, [result_var], span)  # Missing else

    func_body = ir.SeqStmts([if_stmt, ir.ReturnStmt([result_var], span)], span)
    func = ir.Function("test_missing_else", params, return_types, func_body, span)
    program = ir.Program([func], "test_program", span)

    verify_pass = passes.run_verifier()
    with pytest.raises(Exception, match="must have else branch"):
        verify_pass(program)


def test_verify_ssa_valid_control_flow():
    """Test valid control flow passes verification."""
    span = ir.Span.unknown()

    a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
    params: list[ir.Var] = [a]
    return_types: list[ir.Type] = [ir.ScalarType(DataType.INT64)]

    # Valid ForStmt
    loop_var = ir.Var("i", ir.ScalarType(DataType.INDEX), span)
    iter_arg = ir.IterArg("sum", ir.ScalarType(DataType.INT64), a, span)
    yield_value = ir.Add(iter_arg, loop_var, DataType.INT64, span)
    body = ir.YieldStmt([yield_value], span)
    result_var = ir.Var("result", ir.ScalarType(DataType.INT64), span)

    for_stmt = ir.ForStmt(
        loop_var,
        ir.ConstInt(0, DataType.INDEX, span),
        ir.ConstInt(10, DataType.INDEX, span),
        ir.ConstInt(1, DataType.INDEX, span),
        [iter_arg],
        body,
        [result_var],
        span,
    )

    func_body = ir.SeqStmts([for_stmt, ir.ReturnStmt([result_var], span)], span)
    func = ir.Function("test_valid", params, return_types, func_body, span)
    program = ir.Program([func], "test_program", span)

    verify_pass = passes.run_verifier()
    result_program = verify_pass(program)
    assert result_program is not None


class TestConvertToSSAScope:
    """Test SSA conversion is transparent for ScopeStmt."""

    def test_ssa_conversion_transparent_for_scope(self):
        """Test that SSA conversion treats ScopeStmt transparently."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        @pl.program
        class Expected:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        # Apply SSA conversion
        After = passes.convert_to_ssa()(Before)

        # Should be structurally equal (scope is transparent)
        ir.assert_structural_equal(After, Expected)

    def test_ssa_conversion_with_variable_reassignment_in_scope(self):
        """Test SSA conversion renames variables inside scope."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                    y = pl.mul(y, y)
                return y

        # Apply SSA conversion
        After = passes.convert_to_ssa()(Before)

        # Verify SSA pass runs without error
        # The scope should contain renamed variables (y_0, y_1)
        assert After is not None

        # Verify the pass succeeds
        passes.run_verifier()(After)

    def test_ssa_conversion_with_scope_and_outer_code(self):
        """Test SSA conversion with code before and after scope."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                with pl.at(level=pl.Level.CORE_GROUP):
                    b: pl.Tensor[[64], pl.FP32] = pl.mul(a, a)
                c: pl.Tensor[[64], pl.FP32] = pl.add(b, b)
                return c

        # Apply SSA conversion
        After = passes.convert_to_ssa()(Before)

        # Verify SSA pass runs without error
        assert After is not None

        # Verify the pass succeeds
        passes.run_verifier()(After)


class TestScopeViolation:
    """Test SSA scope violation detection."""

    def test_var_used_outside_defining_scope(self):
        """Variable defined inside ForStmt body is not visible after the loop."""
        span = ir.Span.unknown()

        a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
        params: list[ir.Var] = [a]
        return_types: list[ir.Type] = [ir.ScalarType(DataType.INT64)]

        loop_var = ir.Var("i", ir.ScalarType(DataType.INDEX), span)
        inner_var = ir.Var("inner", ir.ScalarType(DataType.INT64), span)
        body = ir.AssignStmt(inner_var, loop_var, span)

        for_stmt = ir.ForStmt(
            loop_var,
            ir.ConstInt(0, DataType.INDEX, span),
            ir.ConstInt(5, DataType.INDEX, span),
            ir.ConstInt(1, DataType.INDEX, span),
            [],
            body,
            [],
            span,
        )

        # inner_var is used outside the loop — scope violation
        func_body = ir.SeqStmts([for_stmt, ir.ReturnStmt([inner_var], span)], span)
        func = ir.Function("test_scope", params, return_types, func_body, span)
        program = ir.Program([func], "test_program", span)

        verify_pass = passes.run_verifier()
        with pytest.raises(Exception, match="used outside its defining scope"):
            verify_pass(program)

    def test_var_from_then_branch_used_after_if(self):
        """Variable defined only in then-branch is not visible after the if."""
        span = ir.Span.unknown()

        a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
        params: list[ir.Var] = [a]
        return_types: list[ir.Type] = [ir.ScalarType(DataType.INT64)]

        condition = ir.Gt(a, ir.ConstInt(0, DataType.INT64, span), DataType.BOOL, span)
        branch_var = ir.Var("branch_only", ir.ScalarType(DataType.INT64), span)
        then_body = ir.AssignStmt(branch_var, a, span)

        if_stmt = ir.IfStmt(condition, then_body, None, [], span)

        # branch_var used after if — scope violation
        func_body = ir.SeqStmts([if_stmt, ir.ReturnStmt([branch_var], span)], span)
        func = ir.Function("test_scope", params, return_types, func_body, span)
        program = ir.Program([func], "test_program", span)

        verify_pass = passes.run_verifier()
        with pytest.raises(Exception, match=r"used before definition|used outside its defining scope"):
            verify_pass(program)

    def test_iter_arg_used_outside_loop(self):
        """IterArg is only visible inside the loop body."""
        span = ir.Span.unknown()

        a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
        params: list[ir.Var] = [a]
        return_types: list[ir.Type] = [ir.ScalarType(DataType.INT64)]

        loop_var = ir.Var("i", ir.ScalarType(DataType.INDEX), span)
        iter_arg = ir.IterArg("acc", ir.ScalarType(DataType.INT64), a, span)
        yield_value = ir.Add(iter_arg, loop_var, DataType.INT64, span)
        body = ir.YieldStmt([yield_value], span)
        result_var = ir.Var("result", ir.ScalarType(DataType.INT64), span)

        for_stmt = ir.ForStmt(
            loop_var,
            ir.ConstInt(0, DataType.INDEX, span),
            ir.ConstInt(5, DataType.INDEX, span),
            ir.ConstInt(1, DataType.INDEX, span),
            [iter_arg],
            body,
            [result_var],
            span,
        )

        # iter_arg used outside the loop — scope violation
        func_body = ir.SeqStmts([for_stmt, ir.ReturnStmt([iter_arg], span)], span)
        func = ir.Function("test_scope", params, return_types, func_body, span)
        program = ir.Program([func], "test_program", span)

        verify_pass = passes.run_verifier()
        with pytest.raises(Exception, match=r"used before definition|used outside its defining scope"):
            verify_pass(program)

    def test_loop_var_used_outside_loop(self):
        """Loop variable is only visible inside the loop body."""
        span = ir.Span.unknown()

        a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
        params: list[ir.Var] = [a]
        return_types: list[ir.Type] = [ir.ScalarType(DataType.INT64)]

        loop_var = ir.Var("i", ir.ScalarType(DataType.INDEX), span)
        body = ir.YieldStmt([], span)

        for_stmt = ir.ForStmt(
            loop_var,
            ir.ConstInt(0, DataType.INDEX, span),
            ir.ConstInt(5, DataType.INDEX, span),
            ir.ConstInt(1, DataType.INDEX, span),
            [],
            body,
            [],
            span,
        )

        # loop_var used outside the loop — scope violation
        func_body = ir.SeqStmts([for_stmt, ir.ReturnStmt([loop_var], span)], span)
        func = ir.Function("test_scope", params, return_types, func_body, span)
        program = ir.Program([func], "test_program", span)

        verify_pass = passes.run_verifier()
        with pytest.raises(Exception, match=r"used before definition|used outside its defining scope"):
            verify_pass(program)

    def test_if_return_var_not_visible_in_condition(self):
        """IfStmt return_vars are only visible after the if finishes."""
        span = ir.Span.unknown()

        a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
        params: list[ir.Var] = [a]
        return_types: list[ir.Type] = [ir.ScalarType(DataType.INT64)]

        result_var = ir.Var("result", ir.ScalarType(DataType.INT64), span)
        # result_var is referenced before the IfStmt defines it via return_vars
        condition = ir.Gt(result_var, ir.ConstInt(0, DataType.INT64, span), DataType.BOOL, span)
        then_body = ir.YieldStmt([a], span)
        else_body = ir.YieldStmt([ir.ConstInt(0, DataType.INT64, span)], span)

        if_stmt = ir.IfStmt(condition, then_body, else_body, [result_var], span)

        func_body = ir.SeqStmts([if_stmt, ir.ReturnStmt([result_var], span)], span)
        func = ir.Function("test_scope", params, return_types, func_body, span)
        program = ir.Program([func], "test_program", span)

        verify_pass = passes.run_verifier()
        with pytest.raises(Exception, match=r"used before definition|used outside its defining scope"):
            verify_pass(program)


class TestCardinalityChecks:
    """Test iter_args/return_vars/yield cardinality mismatch detection."""

    def test_for_iter_args_return_vars_mismatch(self):
        """ForStmt iter_args count != return_vars count."""
        span = ir.Span.unknown()

        a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
        params: list[ir.Var] = [a]
        return_types: list[ir.Type] = [ir.ScalarType(DataType.INT64)]

        loop_var = ir.Var("i", ir.ScalarType(DataType.INDEX), span)
        iter_arg = ir.IterArg("acc", ir.ScalarType(DataType.INT64), a, span)
        body = ir.YieldStmt([iter_arg], span)
        # Two return_vars for one iter_arg — mismatch
        rv1 = ir.Var("r1", ir.ScalarType(DataType.INT64), span)
        rv2 = ir.Var("r2", ir.ScalarType(DataType.INT64), span)

        for_stmt = ir.ForStmt(
            loop_var,
            ir.ConstInt(0, DataType.INDEX, span),
            ir.ConstInt(5, DataType.INDEX, span),
            ir.ConstInt(1, DataType.INDEX, span),
            [iter_arg],
            body,
            [rv1, rv2],
            span,
        )

        func_body = ir.SeqStmts([for_stmt, ir.ReturnStmt([rv1], span)], span)
        func = ir.Function("test_mismatch", params, return_types, func_body, span)
        program = ir.Program([func], "test_program", span)

        verify_pass = passes.run_verifier()
        with pytest.raises(Exception, match=r"(iter_args count.*return_vars count|size mismatch)"):
            verify_pass(program)

    def test_for_yield_count_mismatch(self):
        """ForStmt yield value count != iter_args count."""
        span = ir.Span.unknown()

        a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
        params: list[ir.Var] = [a]
        return_types: list[ir.Type] = [ir.ScalarType(DataType.INT64)]

        loop_var = ir.Var("i", ir.ScalarType(DataType.INDEX), span)
        iter_arg1 = ir.IterArg("acc1", ir.ScalarType(DataType.INT64), a, span)
        iter_arg2 = ir.IterArg("acc2", ir.ScalarType(DataType.INT64), a, span)
        # Only one yield value for two iter_args — mismatch
        body = ir.YieldStmt([iter_arg1], span)
        rv1 = ir.Var("r1", ir.ScalarType(DataType.INT64), span)
        rv2 = ir.Var("r2", ir.ScalarType(DataType.INT64), span)

        for_stmt = ir.ForStmt(
            loop_var,
            ir.ConstInt(0, DataType.INDEX, span),
            ir.ConstInt(5, DataType.INDEX, span),
            ir.ConstInt(1, DataType.INDEX, span),
            [iter_arg1, iter_arg2],
            body,
            [rv1, rv2],
            span,
        )

        func_body = ir.SeqStmts([for_stmt, ir.ReturnStmt([rv1], span)], span)
        func = ir.Function("test_yield_mismatch", params, return_types, func_body, span)
        program = ir.Program([func], "test_program", span)

        verify_pass = passes.run_verifier()
        with pytest.raises(Exception, match=r"(YieldStmt value count.*iter_args count|size mismatch)"):
            verify_pass(program)

    def test_if_yield_count_mismatch(self):
        """IfStmt yield value count != return_vars count."""
        span = ir.Span.unknown()

        a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
        params: list[ir.Var] = [a]
        return_types: list[ir.Type] = [ir.ScalarType(DataType.INT64)]

        condition = ir.Gt(a, ir.ConstInt(0, DataType.INT64, span), DataType.BOOL, span)
        # Two yield values for one return_var — mismatch
        then_body = ir.YieldStmt([a, a], span)
        else_body = ir.YieldStmt([a], span)
        rv = ir.Var("result", ir.ScalarType(DataType.INT64), span)

        if_stmt = ir.IfStmt(condition, then_body, else_body, [rv], span)

        func_body = ir.SeqStmts([if_stmt, ir.ReturnStmt([rv], span)], span)
        func = ir.Function("test_if_yield_mismatch", params, return_types, func_body, span)
        program = ir.Program([func], "test_program", span)

        verify_pass = passes.run_verifier()
        with pytest.raises(Exception, match=r"(YieldStmt value count.*return_vars count|size mismatch)"):
            verify_pass(program)

    # The next four tests construct IR directly via `ir.ForStmt` / `ir.WhileStmt`
    # / `ir.IfStmt` rather than via @pl.program. The DSL parser cannot produce
    # the shape these tests check: it rejects two yield-LHS bindings to the
    # same names (SSA), and a body with two yields that have matching value
    # counts on both yields requires either re-binding LHS names (rejected) or
    # mismatched counts (caught by the structural verifier before SSAVerify
    # runs). The MISPLACED_YIELD check exists as a backstop for IR producers
    # (passes, transforms, IRBuilder users), so direct IR construction is the
    # only way to test it.

    def test_for_mid_body_yield(self):
        """ForStmt body with a YieldStmt followed by another stmt is rejected,
        even when a second YieldStmt sits at the trailing position. The yield
        must be the scope terminator."""
        span = ir.Span.unknown()

        a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
        params: list[ir.Var] = [a]
        return_types: list[ir.Type] = [ir.ScalarType(DataType.INT64)]

        loop_var = ir.Var("i", ir.ScalarType(DataType.INDEX), span)
        iter_arg = ir.IterArg("sum", ir.ScalarType(DataType.INT64), a, span)
        mid_yield = ir.YieldStmt([iter_arg], span)
        trailing_assign = ir.AssignStmt(ir.Var("dummy", ir.ScalarType(DataType.INT64), span), loop_var, span)
        final_yield = ir.YieldStmt([iter_arg], span)
        body = ir.SeqStmts([mid_yield, trailing_assign, final_yield], span)

        rv = ir.Var("result", ir.ScalarType(DataType.INT64), span)
        for_stmt = ir.ForStmt(
            loop_var,
            ir.ConstInt(0, DataType.INDEX, span),
            ir.ConstInt(10, DataType.INDEX, span),
            ir.ConstInt(1, DataType.INDEX, span),
            [iter_arg],
            body,
            [rv],
            span,
        )

        func_body = ir.SeqStmts([for_stmt, ir.ReturnStmt([rv], span)], span)
        func = ir.Function("test_for_mid_yield", params, return_types, func_body, span)
        program = ir.Program([func], "test_program", span)

        verify_pass = passes.run_verifier()
        with pytest.raises(Exception, match="YieldStmt before the terminating position"):
            verify_pass(program)

    def test_while_mid_body_yield(self):
        """WhileStmt body with a YieldStmt followed by another stmt is rejected.
        Mirrors the ForStmt case (both share VerifyLoopIterArgsAndYield)."""
        span = ir.Span.unknown()

        a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
        params: list[ir.Var] = [a]
        return_types: list[ir.Type] = [ir.ScalarType(DataType.INT64)]

        iter_arg = ir.IterArg("acc", ir.ScalarType(DataType.INT64), a, span)
        condition = ir.Gt(iter_arg, ir.ConstInt(0, DataType.INT64, span), DataType.BOOL, span)

        mid_yield = ir.YieldStmt([iter_arg], span)
        trailing_assign = ir.AssignStmt(ir.Var("dummy", ir.ScalarType(DataType.INT64), span), a, span)
        final_yield = ir.YieldStmt([iter_arg], span)
        body = ir.SeqStmts([mid_yield, trailing_assign, final_yield], span)

        rv = ir.Var("result", ir.ScalarType(DataType.INT64), span)
        while_stmt = ir.WhileStmt(condition, [iter_arg], body, [rv], span)

        func_body = ir.SeqStmts([while_stmt, ir.ReturnStmt([rv], span)], span)
        func = ir.Function("test_while_mid_yield", params, return_types, func_body, span)
        program = ir.Program([func], "test_program", span)

        verify_pass = passes.run_verifier()
        with pytest.raises(Exception, match="WhileStmt.*YieldStmt before the terminating position"):
            verify_pass(program)

    def test_if_then_mid_body_yield(self):
        """IfStmt then-branch with a YieldStmt followed by another stmt is
        rejected. Symmetric to the ForStmt case."""
        span = ir.Span.unknown()

        a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
        params: list[ir.Var] = [a]
        return_types: list[ir.Type] = [ir.ScalarType(DataType.INT64)]

        condition = ir.Gt(a, ir.ConstInt(0, DataType.INT64, span), DataType.BOOL, span)
        mid_yield = ir.YieldStmt([a], span)
        trailing_assign = ir.AssignStmt(ir.Var("dummy", ir.ScalarType(DataType.INT64), span), a, span)
        final_yield = ir.YieldStmt([a], span)
        then_body = ir.SeqStmts([mid_yield, trailing_assign, final_yield], span)
        else_body = ir.YieldStmt([a], span)
        rv = ir.Var("result", ir.ScalarType(DataType.INT64), span)

        if_stmt = ir.IfStmt(condition, then_body, else_body, [rv], span)

        func_body = ir.SeqStmts([if_stmt, ir.ReturnStmt([rv], span)], span)
        func = ir.Function("test_if_then_mid_yield", params, return_types, func_body, span)
        program = ir.Program([func], "test_program", span)

        verify_pass = passes.run_verifier()
        with pytest.raises(Exception, match="IfStmt then-branch.*YieldStmt before the terminating position"):
            verify_pass(program)

    def test_if_else_mid_body_yield(self):
        """IfStmt else-branch with a mid-body YieldStmt is rejected. Then-branch
        is well-formed; only the else side fails."""
        span = ir.Span.unknown()

        a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
        params: list[ir.Var] = [a]
        return_types: list[ir.Type] = [ir.ScalarType(DataType.INT64)]

        condition = ir.Gt(a, ir.ConstInt(0, DataType.INT64, span), DataType.BOOL, span)
        then_body = ir.YieldStmt([a], span)
        mid_yield = ir.YieldStmt([a], span)
        trailing_assign = ir.AssignStmt(ir.Var("dummy", ir.ScalarType(DataType.INT64), span), a, span)
        final_yield = ir.YieldStmt([a], span)
        else_body = ir.SeqStmts([mid_yield, trailing_assign, final_yield], span)
        rv = ir.Var("result", ir.ScalarType(DataType.INT64), span)

        if_stmt = ir.IfStmt(condition, then_body, else_body, [rv], span)

        func_body = ir.SeqStmts([if_stmt, ir.ReturnStmt([rv], span)], span)
        func = ir.Function("test_if_else_mid_yield", params, return_types, func_body, span)
        program = ir.Program([func], "test_program", span)

        verify_pass = passes.run_verifier()
        with pytest.raises(Exception, match="IfStmt else-branch.*YieldStmt before the terminating position"):
            verify_pass(program)


class TestValidScopePatterns:
    """Test that valid scope patterns pass verification."""

    def test_return_var_visible_after_for(self):
        """return_vars from ForStmt are visible after the loop."""
        span = ir.Span.unknown()

        a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
        params: list[ir.Var] = [a]
        return_types: list[ir.Type] = [ir.ScalarType(DataType.INT64)]

        loop_var = ir.Var("i", ir.ScalarType(DataType.INDEX), span)
        iter_arg = ir.IterArg("acc", ir.ScalarType(DataType.INT64), a, span)
        yield_value = ir.Add(iter_arg, loop_var, DataType.INT64, span)
        body = ir.YieldStmt([yield_value], span)
        result_var = ir.Var("result", ir.ScalarType(DataType.INT64), span)

        for_stmt = ir.ForStmt(
            loop_var,
            ir.ConstInt(0, DataType.INDEX, span),
            ir.ConstInt(5, DataType.INDEX, span),
            ir.ConstInt(1, DataType.INDEX, span),
            [iter_arg],
            body,
            [result_var],
            span,
        )

        func_body = ir.SeqStmts([for_stmt, ir.ReturnStmt([result_var], span)], span)
        func = ir.Function("test_valid", params, return_types, func_body, span)
        program = ir.Program([func], "test_program", span)

        verify_pass = passes.run_verifier()
        result = verify_pass(program)
        assert result is not None

    def test_return_var_visible_after_if(self):
        """return_vars from IfStmt are visible after the if."""
        span = ir.Span.unknown()

        a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
        params: list[ir.Var] = [a]
        return_types: list[ir.Type] = [ir.ScalarType(DataType.INT64)]

        condition = ir.Gt(a, ir.ConstInt(0, DataType.INT64, span), DataType.BOOL, span)
        then_body = ir.YieldStmt([a], span)
        else_body = ir.YieldStmt([ir.ConstInt(0, DataType.INT64, span)], span)
        result_var = ir.Var("result", ir.ScalarType(DataType.INT64), span)

        if_stmt = ir.IfStmt(condition, then_body, else_body, [result_var], span)

        func_body = ir.SeqStmts([if_stmt, ir.ReturnStmt([result_var], span)], span)
        func = ir.Function("test_valid", params, return_types, func_body, span)
        program = ir.Program([func], "test_program", span)

        verify_pass = passes.run_verifier()
        result = verify_pass(program)
        assert result is not None

    def test_param_visible_everywhere(self):
        """Function parameters are visible in all scopes."""
        span = ir.Span.unknown()

        a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
        params: list[ir.Var] = [a]
        return_types: list[ir.Type] = [ir.ScalarType(DataType.INT64)]

        loop_var = ir.Var("i", ir.ScalarType(DataType.INDEX), span)
        iter_arg = ir.IterArg("acc", ir.ScalarType(DataType.INT64), a, span)
        # Use 'a' (parameter) inside the loop body — should be valid
        yield_value = ir.Add(iter_arg, a, DataType.INT64, span)
        body = ir.YieldStmt([yield_value], span)
        result_var = ir.Var("result", ir.ScalarType(DataType.INT64), span)

        for_stmt = ir.ForStmt(
            loop_var,
            ir.ConstInt(0, DataType.INDEX, span),
            ir.ConstInt(5, DataType.INDEX, span),
            ir.ConstInt(1, DataType.INDEX, span),
            [iter_arg],
            body,
            [result_var],
            span,
        )

        func_body = ir.SeqStmts([for_stmt, ir.ReturnStmt([result_var], span)], span)
        func = ir.Function("test_valid", params, return_types, func_body, span)
        program = ir.Program([func], "test_program", span)

        verify_pass = passes.run_verifier()
        result = verify_pass(program)
        assert result is not None

    def test_dsl_ssa_converted_passes_scope_check(self):
        """DSL program converted to SSA passes scope verification."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                for i in pl.range(5):
                    result = pl.mul(result, result)
                return result

        After = passes.convert_to_ssa()(Before)
        passes.run_verifier()(After)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
