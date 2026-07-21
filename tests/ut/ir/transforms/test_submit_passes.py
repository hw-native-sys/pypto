# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for IR passes operating on Submit nodes.

The parser emits ``ir.Submit`` for ``pl.submit(...)``. These tests
construct Submit-bearing IR directly (bypassing the DSL) and verify that
DCE / SSA and the printer's round-trip preserve the structural shape
(op, args, first-class deps_) without leaking Vars or degrading Submit
to Call.
"""

import pypto.language as pl
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


def _build_program_with_spmd_submit(core_num_is_var: bool = False) -> ir.Program:
    """Build a caller that ``pl.spmd_submit``s a kernel (Submit + launch spec).

    When ``core_num_is_var`` the launch ``core_num`` references the (reassigned)
    caller arg so SSA conversion must remap it — exercising the IRMutator's
    first-class ``core_num_`` walk. Otherwise ``core_num`` is a constant.
    """
    span = ir.Span.unknown()
    kernel_x = ir.Var("x", ir.ScalarType(DataType.INDEX), span)
    kernel = ir.Function(
        "kernel", [kernel_x], [ir.ScalarType(DataType.INDEX)], ir.ReturnStmt([kernel_x], span), span
    )
    kernel_gvar = ir.GlobalVar("kernel")

    caller_arg = ir.Var("a", ir.ScalarType(DataType.INDEX), span)
    tid_arg = ir.Var("t", ir.ScalarType(DataType.TASK_ID), span)
    submit_ret_ty = ir.TupleType([ir.ScalarType(DataType.INDEX), ir.ScalarType(DataType.TASK_ID)])
    res_var = ir.Var("res", submit_ret_ty, span)

    stmts: list[ir.Stmt] = []
    if core_num_is_var:
        # Reassign `a` so SSA mints a fresh version; core_num references it.
        one = ir.ConstInt(1, DataType.INDEX, span)
        stmts.append(ir.AssignStmt(caller_arg, ir.Add(caller_arg, one, DataType.INDEX, span), span))
        core_num: ir.Expr = caller_arg
    else:
        core_num = ir.ConstInt(4, DataType.INDEX, span)

    submit = ir.Submit(
        kernel_gvar,
        [caller_arg],
        [tid_arg],
        {},
        None,
        submit_ret_ty,
        span,
        core_num=core_num,
        sync_start=True,
    )
    stmts.append(ir.AssignStmt(res_var, submit, span))
    stmts.append(ir.ReturnStmt([res_var], span))

    caller = ir.Function("caller", [caller_arg, tid_arg], [submit_ret_ty], ir.SeqStmts(stmts, span), span)
    return ir.Program([kernel, caller], "spmd_submit_smoke", span)


def test_ssa_preserves_spmd_submit_launch_spec():
    """convert_to_ssa() must carry the SPMD launch spec (core_num / sync_start)
    through the Submit reconstruction — a pass that dropped them would silently
    downgrade an SPMD launch to a single-block submit."""
    program_after = passes.convert_to_ssa()(_build_program_with_spmd_submit(core_num_is_var=False))
    caller_after = program_after.get_function("caller")
    assert caller_after is not None
    submit_after = _find_submit_in_function(caller_after)
    assert submit_after is not None
    assert submit_after.sync_start is True
    assert isinstance(submit_after.core_num, ir.ConstInt)
    assert submit_after.core_num.value == 4


def test_ssa_remaps_spmd_submit_core_num_var():
    """When core_num references a Var that SSA renames, the rebuilt Submit's
    core_num must point at the fresh version (IRMutator walks core_num_)."""
    program_after = passes.convert_to_ssa()(_build_program_with_spmd_submit(core_num_is_var=True))
    caller_after = program_after.get_function("caller")
    assert caller_after is not None
    submit_after = _find_submit_in_function(caller_after)
    assert submit_after is not None
    assert submit_after.core_num is not None
    core_num_var = submit_after.core_num
    assert isinstance(core_num_var, ir.Var)
    # The original `a` parameter was reassigned; core_num must reference the
    # latest SSA version, not the stale parameter.
    assert core_num_var is not list(caller_after.params)[0]
    # And it must be the same Var the (remapped) arg references.
    assert core_num_var is submit_after.args[0]


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


# ---------------------------------------------------------------------------
# Scope-form dispatch predicate — ``with pl.spmd(..., predicate=...)``
# ---------------------------------------------------------------------------
#
# ``pl.spmd_submit(..., predicate=)`` builds a Submit at parse time, so the
# predicate is covered by the Submit field walk above. The *scope* form is
# outlined only later, so between parse and outline the predicate lives on
# ``SpmdScopeStmt.attrs`` as an Expr carrying live SSA Vars (the operand tensor
# and its indices).
#
# Three code paths must know about that attr — IRVisitor::VisitScopeAttrs,
# IRMutator::MutateScopeAttrs, and ConvertToSSA::SubstScopeAttrs. Missing the
# SSA one leaves the predicate pointing at the pre-SSA Var: the IR still
# verifies and codegen still emits a predicate, but it reads a *dangling*
# operand. These tests pin the observable consequence rather than the
# mechanism.

_SCOPE_PREDICATE_PROGRAM = """
import pypto.language as pl


@pl.program
class Prog:
    @pl.function(type=pl.FunctionType.InCore)
    def expert(
        self, x: pl.Tensor[[512, 128], pl.FP32], out: pl.Out[pl.Tensor[[512, 128], pl.FP32]]
    ) -> pl.Tensor[[512, 128], pl.FP32]:
        t = pl.load(x, [0, 0], [128, 128])
        out = pl.store(t, [0, 0], out)
        return out

    @pl.function(type=pl.FunctionType.InCore)
    def gate(
        self, g: pl.Out[pl.Tensor[[512, 128], pl.INT32]]
    ) -> pl.Tensor[[512, 128], pl.INT32]:
        t = pl.load(g, [0, 0], [128, 128])
        g = pl.store(t, [0, 0], g)
        return g

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        x: pl.Tensor[[512, 128], pl.FP32],
        out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
        rc: pl.Out[pl.Tensor[[512, 128], pl.INT32]],
    ) -> pl.Tensor[[512, 128], pl.FP32]:
        with pl.spmd(1) as g_tid:
            rc = self.gate(rc)
        with pl.spmd(1, deps=[g_tid], predicate=(rc[0, 0] > 0)) as t:
            out = self.expert(x, out)
        return out
"""


def _spmd_scopes(program: ir.Program) -> list:
    """Every SpmdScopeStmt in ``main``, in source order."""
    found: list = []

    def walk(stmt) -> None:
        if isinstance(stmt, ir.SpmdScopeStmt):
            found.append(stmt)
        for field in ("stmts", "body"):
            value = getattr(stmt, field, None)
            if isinstance(value, list):
                for child in value:
                    walk(child)
            elif value is not None:
                walk(value)

    main = program.get_function("main")
    assert main is not None
    walk(main.body)
    return found


def _predicate_operand(scope) -> ir.Var:
    """The tensor Var a scope's ``predicate`` attr reads."""
    predicate = dict(scope.attrs.items())["predicate"]
    operand = predicate.left
    while isinstance(operand, ir.Cast):
        operand = operand.operand
    assert isinstance(operand, ir.Call)  # tensor.read
    tensor = operand.args[0]
    assert isinstance(tensor, ir.Var)
    return tensor


def test_ssa_renames_scope_predicate_operand():
    """The operand Var inside the scope-attr Expr is versioned like any other use.

    ``rc`` is rebound by the gate scope, so SSA must rewrite the predicate's
    operand to the post-rebind version. Without the ConvertToSSA scope-attr
    branch it would keep pointing at the original parameter Var.
    """
    program = pl.parse_program(_SCOPE_PREDICATE_PROGRAM)
    before = _predicate_operand(_spmd_scopes(program)[1])
    assert before.name_hint == "rc"

    after_program = passes.convert_to_ssa()(program)
    after = _predicate_operand(_spmd_scopes(after_program)[1])

    assert after.unique_id != before.unique_id, "predicate operand was not SSA-versioned"
    assert after.name_hint.startswith("rc"), after.name_hint


def test_ssa_predicate_operand_matches_the_gate_scope_result():
    """The versioned operand is the value the gate scope produced, not a fresh Var.

    Renaming to *some* new Var would satisfy the test above; this pins that it
    renames to the same SSA version the producing scope defines, which is what
    makes the dispatch-point read observe the current value.
    """
    program = passes.convert_to_ssa()(pl.parse_program(_SCOPE_PREDICATE_PROGRAM))
    gate_scope, expert_scope = _spmd_scopes(program)
    operand = _predicate_operand(expert_scope)

    produced = [
        stmt.var.unique_id
        for stmt in _flatten_stmts(gate_scope.body)
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.var, ir.Var)
    ]
    assert operand.unique_id in produced, (
        "predicate operand should be the gate scope's SSA result, "
        f"got {operand.name_hint} (#{operand.unique_id}), scope defines {produced}"
    )


def _flatten_stmts(stmt) -> list:
    """All statements under ``stmt``, flattening SeqStmts and scope bodies."""
    out: list = []

    def walk(node) -> None:
        out.append(node)
        for field in ("stmts", "body"):
            value = getattr(node, field, None)
            if isinstance(value, list):
                for child in value:
                    walk(child)
            elif value is not None:
                walk(value)

    walk(stmt)
    return out


def test_structural_hash_handles_var_and_expr_scope_attrs():
    """Scopes carrying Var-/Expr-valued attrs must be hashable.

    ``structural_hash``'s attr codec used to throw on any attr it could not
    hash, which made every ``with pl.spmd(...) as tid:`` (``task_id_var``) and
    every ``with pl.spmd(..., predicate=...)`` un-hashable — valid IR that a
    caching or dedup path would crash on. Such attrs are now skipped (the hash
    is coarser; ``structural_equal`` still distinguishes them).
    """
    program = pl.parse_program(_SCOPE_PREDICATE_PROGRAM)
    assert isinstance(ir.structural_hash(program), int)

    # Equal programs still hash equal.
    again = pl.parse_program(_SCOPE_PREDICATE_PROGRAM)
    assert ir.structural_hash(program) == ir.structural_hash(again)

    # ...and structural_equal remains the authority on the predicate itself.
    no_predicate = pl.parse_program(_SCOPE_PREDICATE_PROGRAM.replace(", predicate=(rc[0, 0] > 0)", ""))
    assert isinstance(ir.structural_hash(no_predicate), int)
    assert not ir.structural_equal(program, no_predicate)


# A predicate whose index is a *computed* Var (``idx``), used nowhere else.
# That makes the assignment feeding it dead unless DCE counts the predicate as a
# use — see test_dce_keeps_the_predicate_operands_producer.
_SCOPE_PREDICATE_LIVE_INDEX_PROGRAM = """
import pypto.language as pl


@pl.program
class Prog:
    @pl.function(type=pl.FunctionType.InCore)
    def expert(
        self, x: pl.Tensor[[512, 128], pl.FP32], out: pl.Out[pl.Tensor[[512, 128], pl.FP32]]
    ) -> pl.Tensor[[512, 128], pl.FP32]:
        t = pl.load(x, [0, 0], [128, 128])
        out = pl.store(t, [0, 0], out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        x: pl.Tensor[[512, 128], pl.FP32],
        out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
        rc: pl.Tensor[[512, 128], pl.INT32],
        i: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[512, 128], pl.FP32]:
        idx = i + 1
        with pl.spmd(1, predicate=(rc[idx, 0] > 0)):
            out = self.expert(x, out)
        return out
"""


def test_dce_keeps_the_predicate_operands_producer():
    """A Var used *only* inside the predicate is a live use, not dead code.

    ``idx`` feeds nothing but the predicate's index. Without the predicate
    branch in DCE's scope-attr live-root collection, its assignment is deleted
    and the attr is left referencing a free variable — the IR still prints and
    passes structural checks, so nothing else catches it. Regression for the
    same failure class as issue #1456.
    """
    program = passes.simplify()(pl.parse_program(_SCOPE_PREDICATE_LIVE_INDEX_PROGRAM))
    main = program.get_function("main")
    assert main is not None
    printed = ir.python_print(main)

    assert "idx" in printed, printed
    # The tell-tale of a dropped live root: the printer renders an undefined
    # reference with a __FREE_VAR suffix.
    assert "__FREE_VAR" not in printed, f"predicate references a dangling Var:\n{printed}"
    # The assignment itself survived, not just the name inside the predicate.
    assert "idx: pl.Scalar" in printed, printed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
