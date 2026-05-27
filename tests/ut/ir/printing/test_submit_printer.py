# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for printing and structural identity of the new Submit IR node."""

import pytest
from pypto import DataType, ir


def _make_kernel_function(name: str) -> tuple[ir.GlobalVar, ir.Function]:
    """Build a trivial kernel function ``def <name>(x: INDEX) -> INDEX: yield_ x``."""
    span = ir.Span.unknown()
    x = ir.Var("x", ir.ScalarType(DataType.INDEX), span)
    body = ir.YieldStmt([x], span)
    func = ir.Function(name, [x], [ir.ScalarType(DataType.INDEX)], body, span)
    return ir.GlobalVar(name), func


def _make_program_with_submit(deps: list[ir.Expr]) -> tuple[ir.Program, ir.Submit]:
    """Build a Program containing a Submit that targets a kernel function.

    Returns the Program (so the printer enters a Program context) and the
    Submit node so the test can inspect it directly.
    """
    span = ir.Span.unknown()
    kernel_gvar, kernel_func = _make_kernel_function("kernel")

    arg = ir.Var("a", ir.ScalarType(DataType.INDEX), span)
    submit = ir.Submit(
        kernel_gvar,
        [arg],
        deps,
        ir.TupleType([ir.ScalarType(DataType.INDEX), ir.ScalarType(DataType.TASK_ID)]),
        span,
    )
    lhs = ir.Var("res", ir.TupleType([ir.ScalarType(DataType.INDEX), ir.ScalarType(DataType.TASK_ID)]), span)
    body = ir.SeqStmts([ir.AssignStmt(lhs, submit, span), ir.YieldStmt([lhs], span)], span)
    caller_func = ir.Function(
        "caller",
        [arg],
        [ir.TupleType([ir.ScalarType(DataType.INDEX), ir.ScalarType(DataType.TASK_ID)])],
        body,
        span,
    )
    program = ir.Program([kernel_func, caller_func], "test_submit_program", span)
    return program, submit


def test_submit_prints_as_pl_submit_inside_program():
    program, _ = _make_program_with_submit(deps=[])
    text = program.as_python()
    assert "pl.submit(self.kernel, a)" in text, text


def test_submit_prints_deps_kwarg():
    """Deps bound as function parameters print as bare names (no __FREE_VAR suffix)."""
    span = ir.Span.unknown()
    kernel_gvar, kernel_func = _make_kernel_function("kernel")

    arg = ir.Var("a", ir.ScalarType(DataType.INDEX), span)
    t1 = ir.Var("t1", ir.ScalarType(DataType.TASK_ID), span)
    t2 = ir.Var("t2", ir.ScalarType(DataType.TASK_ID), span)
    submit = ir.Submit(
        kernel_gvar,
        [arg],
        [t1, t2],
        ir.TupleType([ir.ScalarType(DataType.INDEX), ir.ScalarType(DataType.TASK_ID)]),
        span,
    )
    lhs = ir.Var("res", ir.TupleType([ir.ScalarType(DataType.INDEX), ir.ScalarType(DataType.TASK_ID)]), span)
    body = ir.SeqStmts([ir.AssignStmt(lhs, submit, span), ir.YieldStmt([lhs], span)], span)
    caller_func = ir.Function(
        "caller",
        [arg, t1, t2],
        [ir.TupleType([ir.ScalarType(DataType.INDEX), ir.ScalarType(DataType.TASK_ID)])],
        body,
        span,
    )
    program = ir.Program([kernel_func, caller_func], "test_submit_program", span)
    text = program.as_python()
    assert "pl.submit(self.kernel, a, deps=[t1, t2])" in text, text


def test_submit_zero_arg_with_deps():
    """A Submit with no positional args still emits deps=[...] cleanly (no dangling comma)."""
    span = ir.Span.unknown()
    kernel_gvar, kernel_func = _make_kernel_function("seed")
    t = ir.Var("t", ir.ScalarType(DataType.TASK_ID), span)
    submit = ir.Submit(
        kernel_gvar,
        [],
        [t],
        ir.TupleType([ir.ScalarType(DataType.TASK_ID)]),
        span,
    )
    lhs = ir.Var("res", ir.TupleType([ir.ScalarType(DataType.TASK_ID)]), span)
    body = ir.SeqStmts([ir.AssignStmt(lhs, submit, span), ir.YieldStmt([lhs], span)], span)
    caller_func = ir.Function("caller", [t], [ir.TupleType([ir.ScalarType(DataType.TASK_ID)])], body, span)
    program = ir.Program([kernel_func, caller_func], "test_submit_zero_arg", span)
    text = program.as_python()
    assert "pl.submit(self.seed, deps=[t])" in text, text


def test_submit_structural_equal_self():
    """A Submit is structurally equal to itself."""
    _, submit_a = _make_program_with_submit(deps=[])
    assert ir.structural_equal(submit_a, submit_a)
    assert ir.structural_hash(submit_a) == ir.structural_hash(submit_a)


def test_submit_structural_inequal_to_call_with_same_args():
    """A Submit must NOT compare structurally equal to a Call, even with identical op/args.

    This is the whole point of the new IR node — it's structurally distinct from
    a plain function Call.
    """
    span = ir.Span.unknown()
    kernel_gvar = ir.GlobalVar("kernel")
    arg = ir.Var("a", ir.ScalarType(DataType.INDEX), span)

    submit = ir.Submit(
        kernel_gvar,
        [arg],
        [],
        ir.TupleType([ir.ScalarType(DataType.INDEX), ir.ScalarType(DataType.TASK_ID)]),
        span,
    )
    call = ir.Call(kernel_gvar, [arg], ir.ScalarType(DataType.INDEX), span)

    assert not ir.structural_equal(submit, call)


def test_submit_structural_equal_is_deps_order_sensitive():
    """Two Submits with the same deps in the same order are equal; swapping
    the order makes them unequal — deps are positional, not a set. Confirm
    hash equality matches for the same-order case.
    """
    span = ir.Span.unknown()
    kernel_gvar = ir.GlobalVar("kernel")
    t1 = ir.Var("t1", ir.ScalarType(DataType.TASK_ID), span)
    t2 = ir.Var("t2", ir.ScalarType(DataType.TASK_ID), span)

    submit_a = ir.Submit(kernel_gvar, [], [t1, t2], ir.ScalarType(DataType.TASK_ID), span)
    submit_b = ir.Submit(kernel_gvar, [], [t1, t2], ir.ScalarType(DataType.TASK_ID), span)
    assert ir.structural_equal(submit_a, submit_b)
    assert ir.structural_hash(submit_a) == ir.structural_hash(submit_b)

    submit_swapped = ir.Submit(kernel_gvar, [], [t2, t1], ir.ScalarType(DataType.TASK_ID), span)
    assert not ir.structural_equal(submit_a, submit_swapped)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
