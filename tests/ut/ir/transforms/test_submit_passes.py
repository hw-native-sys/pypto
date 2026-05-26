# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for IR passes operating on Submit nodes.

The parser does not yet emit Submit (Phase 3); these tests construct
Submit-bearing IR directly and verify that the passes that have been
migrated do not crash on Submit and preserve its structural shape (op,
args, first-class deps_).
"""

import pytest
from pypto import DataType, ir, passes


def _build_program_with_submit(reassign: bool = False) -> ir.Program:
    """Build a Program with one kernel and a caller that pl.submits it.

    When ``reassign`` is True the caller reassigns a Var so SSA conversion
    has actual work to do (otherwise the input is already in SSA form and
    the pass is a no-op).
    """
    span = ir.Span.unknown()
    kernel_x = ir.Var("x", ir.ScalarType(DataType.INDEX), span)
    kernel = ir.Function(
        "kernel",
        [kernel_x],
        [ir.ScalarType(DataType.INDEX)],
        ir.ReturnStmt([kernel_x], span),
        span,
    )
    kernel_gvar = ir.GlobalVar("kernel")

    caller_arg = ir.Var("a", ir.ScalarType(DataType.INDEX), span)
    tid_arg = ir.Var("t", ir.ScalarType(DataType.TASK_ID), span)
    submit_ret_ty = ir.TupleType([ir.ScalarType(DataType.INDEX), ir.ScalarType(DataType.TASK_ID)])
    res_var = ir.Var("res", submit_ret_ty, span)

    stmts: list[ir.Stmt] = []
    if reassign:
        # Reassign caller_arg so SSA conversion mints a fresh version that the
        # Submit's args reference. After SSA the Submit's args_[0] should point
        # to the latest version of `a`.
        one = ir.ConstInt(1, DataType.INDEX, span)
        stmts.append(ir.AssignStmt(caller_arg, ir.Add(caller_arg, one, DataType.INDEX, span), span))

    submit = ir.Submit(kernel_gvar, [caller_arg], [tid_arg], submit_ret_ty, span)
    stmts.append(ir.AssignStmt(res_var, submit, span))
    stmts.append(ir.ReturnStmt([res_var], span))

    body = ir.SeqStmts(stmts, span)
    caller = ir.Function("caller", [caller_arg, tid_arg], [submit_ret_ty], body, span)
    return ir.Program([kernel, caller], "submit_pipeline_smoke", span)


def _find_submit_in_function(func: ir.Function) -> ir.Submit | None:
    """Return the first Submit node in ``func``'s body, or None."""
    body = func.body
    if isinstance(body, ir.SeqStmts):
        stmts = list(body.stmts)
    else:
        stmts = [body]
    for stmt in stmts:
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Submit):
            return stmt.value
    return None


def test_ssa_preserves_submit_node_kind():
    """convert_to_ssa() must preserve Submit-ness — the result still has a
    Submit on the assignment RHS, not a degraded plain Call. Default
    VerificationLevel.BASIC enables the print → re-parse round-trip
    instrument, which now accepts the single-LHS Submit print form."""
    program_before = _build_program_with_submit(reassign=False)
    program_after = passes.convert_to_ssa()(program_before)

    caller_after = program_after.get_function("caller")
    assert caller_after is not None
    submit_after = _find_submit_in_function(caller_after)
    assert submit_after is not None, "SSA pass must keep the Submit; got body without one"
    assert isinstance(submit_after, ir.Submit)
    assert len(submit_after.args) == 1
    assert len(submit_after.deps) == 1


def test_ssa_renames_submit_args_and_deps():
    """When SSA conversion mints a fresh version of a Var that the Submit
    references in args or deps, the rebuilt Submit must reference the new
    version (verifies the IRMutator default walks both fields)."""
    program_before = _build_program_with_submit(reassign=True)
    program_after = passes.convert_to_ssa()(program_before)

    caller_after = program_after.get_function("caller")
    assert caller_after is not None
    submit_after = _find_submit_in_function(caller_after)
    assert submit_after is not None

    # The reassigned arg `a` was rewritten by SSA — the Submit's args[0]
    # must point at the latest SSA version, not the original `a` parameter.
    arg_var = submit_after.args[0]
    assert isinstance(arg_var, ir.Var)
    caller_params = list(caller_after.params)
    assert arg_var is not caller_params[0]


def test_submit_round_trips_through_ssa():
    """An SSA-converted Submit-bearing program prints the pl.submit form."""
    program_before = _build_program_with_submit(reassign=False)
    program_after = passes.convert_to_ssa()(program_before)

    text = program_after.as_python()
    assert "pl.submit(self.kernel" in text, text


def test_submit_single_lhs_form_round_trips():
    """The single-LHS print form ``res: pl.Tuple[..., TASK_ID] = pl.submit(...)``
    is re-accepted by the parser, which means
    ``passes.convert_to_ssa()`` with default ``VerificationLevel.BASIC``
    (round-trip enabled) accepts a Submit-bearing program. Regression
    guard against the parser-side fix.
    """
    program_before = _build_program_with_submit(reassign=False)
    # No explicit PassContext — default verification is BASIC, which runs
    # the RoundtripInstrument on every pass. If the parser had still
    # required ``out, tid = ...`` unpacking, this call would raise.
    passes.convert_to_ssa()(program_before)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
