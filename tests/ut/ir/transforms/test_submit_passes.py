# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for IR passes operating on Submit nodes.

The parser emits ``ir.Submit`` for ``pl.submit(...)`` / ``pl.spmd_submit(...)``,
so every program here is DSL-authored in the Before/Expected style. The
comparisons verify that DCE / SSA preserve the structural shape (op, args,
first-class ``deps_``, the SPMD launch spec) without leaking Vars or degrading
Submit to Call.

The print -> re-parse round-trip of the single-LHS ``pl.submit`` form is
asserted explicitly by ``test_submit_single_lhs_form_round_trips`` below,
NOT left to the ambient verification fixture. The conftest installs the
round-trip instrument only when ``PYPTO_VERIFY_LEVEL`` is ``roundtrip`` (the
default); under the supported ``basic`` level it installs property
verification alone, so the structural comparisons here would never reach the
printer or the parser.
"""

import pypto.language as pl
import pytest
from pypto import ir, passes

# ---------------------------------------------------------------------------
# ConvertToSSA over ``pl.submit`` / ``pl.spmd_submit``.
#
# Each pair below is Before/Expected in the DSL. ``assert_structural_equal``
# maps Vars by definition site rather than by name, so ``Expected`` spells the
# post-SSA versions with readable names (``a_1``) instead of the printer's
# ``a__ssa_v1``. A single structural comparison covers every property these
# tests exist for at once: the RHS is still a ``Submit`` (a degraded plain
# ``Call`` has a different node kind), ``args`` / ``deps`` / ``core_num`` point
# at the post-rename versions, and the SPMD launch spec survives.
#
# The single-LHS ``pl.submit`` print form is pinned separately and explicitly
# by ``test_submit_single_lhs_form_round_trips``; these comparisons are
# structural only and do not exercise the printer.
# ---------------------------------------------------------------------------


def test_ssa_preserves_submit_node_kind():
    """An already-SSA Submit survives the pass unchanged — still a Submit, with
    its single arg and its first-class dep intact."""

    @pl.program
    class Before:
        @pl.function
        def kernel(self, x: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
            return x

        @pl.function
        def caller(self, a: pl.Scalar[pl.INDEX], t: pl.Scalar[pl.TASK_ID]):
            with pl.manual_scope():
                res, tid = pl.submit(self.kernel, a, deps=[t])
            return res, tid

    ir.assert_structural_equal(passes.convert_to_ssa()(Before), Before)


def test_ssa_renames_submit_args_and_deps():
    """A reassigned Var reaches the Submit's ``args`` as its latest version.

    ``a`` is rebound before the submit, so the rebuilt Submit must reference the
    post-rebind value rather than the original parameter — which is what pins
    that the IRMutator default walks both ``args_`` and ``deps_``.
    """

    @pl.program
    class Before:
        @pl.function
        def kernel(self, x: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
            return x

        @pl.function
        def caller(self, a: pl.Scalar[pl.INDEX], t: pl.Scalar[pl.TASK_ID]):
            a = a + 1
            with pl.manual_scope():
                res, tid = pl.submit(self.kernel, a, deps=[t])
            return res, tid

    @pl.program
    class Expected:
        @pl.function(strict_ssa=True)
        def kernel(self, x: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
            return x

        @pl.function(strict_ssa=True)
        def caller(self, a: pl.Scalar[pl.INDEX], t: pl.Scalar[pl.TASK_ID]):
            a_1 = a + 1
            with pl.manual_scope():
                res, tid = pl.submit(self.kernel, a_1, deps=[t])
            return res, tid

    ir.assert_structural_equal(passes.convert_to_ssa()(Before), Expected)


def test_ssa_preserves_spmd_submit_launch_spec():
    """The SPMD launch spec (``core_num`` / ``sync_start``) survives the Submit
    reconstruction — a pass that dropped it would silently downgrade an SPMD
    launch to a single-block submit."""

    @pl.program
    class Before:
        @pl.function
        def kernel(self, x: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
            return x

        @pl.function
        def caller(self, a: pl.Scalar[pl.INDEX], t: pl.Scalar[pl.TASK_ID]):
            with pl.manual_scope():
                res, tid = pl.spmd_submit(self.kernel, a, core_num=4, sync_start=True, deps=[t])
            return res, tid

    ir.assert_structural_equal(passes.convert_to_ssa()(Before), Before)


def test_ssa_remaps_spmd_submit_core_num_var():
    """A ``core_num`` that reads a renamed Var is remapped like any other use.

    ``core_num`` is a first-class Submit field rather than an ordinary arg, so
    it needs its own IRMutator walk; ``Expected`` pins that it lands on the same
    post-rebind version the arg does.
    """

    @pl.program
    class Before:
        @pl.function
        def kernel(self, x: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
            return x

        @pl.function
        def caller(self, a: pl.Scalar[pl.INDEX], t: pl.Scalar[pl.TASK_ID]):
            a = a + 1
            with pl.manual_scope():
                res, tid = pl.spmd_submit(self.kernel, a, core_num=a, sync_start=True, deps=[t])
            return res, tid

    @pl.program
    class Expected:
        @pl.function(strict_ssa=True)
        def kernel(self, x: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
            return x

        @pl.function(strict_ssa=True)
        def caller(self, a: pl.Scalar[pl.INDEX], t: pl.Scalar[pl.TASK_ID]):
            a_1 = a + 1
            with pl.manual_scope():
                res, tid = pl.spmd_submit(self.kernel, a_1, core_num=a_1, sync_start=True, deps=[t])
            return res, tid

    ir.assert_structural_equal(passes.convert_to_ssa()(Before), Expected)


def test_submit_single_lhs_form_round_trips():
    """The single-LHS print form re-parses to a structurally identical program.

    ``convert_to_ssa`` prints a Submit as
    ``res: pl.Tuple[..., pl.Scalar[pl.TASK_ID]] = pl.submit(...)`` rather than
    as an ``out, tid = ...`` unpacking, and the parser has to accept that form
    back. Asserted here with an explicit print -> parse -> compare so it holds
    at every ``PYPTO_VERIFY_LEVEL``, including ``basic``, where the conftest
    installs no round-trip instrument.
    """

    @pl.program
    class Before:
        @pl.function
        def kernel(self, x: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
            return x

        @pl.function
        def caller(self, a: pl.Scalar[pl.INDEX], t: pl.Scalar[pl.TASK_ID]):
            with pl.manual_scope():
                res, tid = pl.submit(self.kernel, a, deps=[t])
            return res, tid

    After = passes.convert_to_ssa()(Before)
    printed = After.as_python()
    assert "pl.submit(self.kernel" in printed, printed
    ir.assert_structural_equal(pl.parse_program(printed), After)


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
    return _operand_of(dict(scope.attrs.items())["predicate"])


def _operand_of(predicate) -> ir.Var:
    """The tensor Var a predicate comparison reads, in either operand order.

    Shared by the scope-attr form (before outlining) and the ``Submit.predicate``
    field form (after), so both assert against the same extraction.
    """
    operand = predicate.right if isinstance(predicate.left, ir.ConstInt) else predicate.left
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


# ---------------------------------------------------------------------------
# Scope-form dispatch predicate — ``with pl.at(level=CORE_GROUP, predicate=...)``
# ---------------------------------------------------------------------------
#
# The pl.at form rides the same scope-attr rail as pl.spmd, one pass earlier
# (OutlineIncoreScopes rather than OutlineClusterScopes). What is genuinely
# different is *who produces the operand*: a pl.at body typically writes the
# tensor with ``pl.store``, which the outliner exports under a fresh call-site
# Var. The attr still names the scope-local post-store alias, so the outliner
# must resolve it through ``store_target_renames_`` exactly as it does for the
# synthesised call's args — otherwise the emitted Submit reads a Var that no
# longer exists in the parent function.

_AT_PREDICATE_PROGRAM = """
import pypto.language as pl


@pl.program
class Prog:
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        x: pl.Tensor[[512, 128], pl.FP32],
        out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
        rc: pl.Out[pl.Tensor[[512, 128], pl.INT32]],
    ) -> pl.Tensor[[512, 128], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP) as g_tid:
            g = pl.load(rc, [0, 0], [128, 128])
            rc = pl.store(g, [0, 0], rc)
        with pl.at(level=pl.Level.CORE_GROUP, deps=[g_tid], predicate=(rc[0, 0] > 0)) as t:
            v = pl.load(x, [0, 0], [128, 128])
            out = pl.store(v, [0, 0], out)
        return out
"""


def _incore_scopes(program: ir.Program) -> list:
    """Every InCoreScopeStmt in ``main``, in source order."""
    found: list = []

    def walk(stmt) -> None:
        if isinstance(stmt, ir.InCoreScopeStmt):
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


def test_ssa_renames_at_scope_predicate_operand():
    """The operand Var inside a pl.at scope attr is versioned like any other use."""
    program = pl.parse_program(_AT_PREDICATE_PROGRAM)
    before = _predicate_operand(_incore_scopes(program)[1])
    assert before.name_hint == "rc"

    after_program = passes.convert_to_ssa()(program)
    after = _predicate_operand(_incore_scopes(after_program)[1])

    assert after.unique_id != before.unique_id, "predicate operand was not SSA-versioned"
    assert after.name_hint.startswith("rc"), after.name_hint


def test_outlining_rebinds_a_store_produced_predicate_operand():
    """The outliner must resolve the operand to the value current at this scope.

    The gate scope writes ``rc`` via ``pl.store``, so after SSA the predicate
    names the scope-*local* post-store alias. ``OutlineIncoreScopes`` exports
    that store target under a fresh call-site Var and drops the alias, so an
    unresolved predicate would leave the emitted ``Submit`` reading a Var with
    no definition in ``main`` (printed with a ``__FREE_VAR`` suffix, and caught
    by UseAfterDefCheck once verification runs).
    """
    program = pl.parse_program(_AT_PREDICATE_PROGRAM)
    for factory in (
        passes.inline_functions,
        passes.unroll_loops,
        passes.ctrl_flow_transform,
        passes.convert_to_ssa,
        passes.simplify,
        passes.normalize_stmt_structure,
        passes.flatten_call_expr,
        passes.outline_hierarchy_scopes,
        passes.outline_incore_scopes,
    ):
        program = factory()(program)

    main = program.get_function("main")
    assert main is not None
    printed = ir.python_print(main)
    assert "predicate=(" in printed, printed
    assert "__FREE_VAR" not in printed, f"predicate references a dangling Var:\n{printed}"

    # A dangling operand is only half the failure mode: rebinding to the *input*
    # ``rc`` would also be defined and also print without ``__FREE_VAR``, while
    # making the dispatch read the pre-gate value. Pin the operand to the result
    # the gate Submit actually exports.
    gate, expert = _submits(main.body)
    assert expert.predicate is not None, ir.python_print(main)
    operand = _submit_predicate_operand(expert)

    # SSA has versioned the parameter by now (``rc`` -> ``rc__ssa_v0``).
    rc_param = next(param for param in main.params if param.name_hint.startswith("rc"))
    assert operand.unique_id != rc_param.unique_id, (
        f"predicate operand is the pre-gate input rc, so the dispatch would read a stale value:\n{printed}"
    )
    exported = _tuple_result_vars(main.body, gate)
    assert operand.unique_id in exported, (
        "predicate operand should be the rc result exported by the gate submit, "
        f"got {operand.name_hint} (#{operand.unique_id}); gate exports {exported}\n{printed}"
    )


def _submits(body) -> list:
    """Every ``Submit`` bound by an AssignStmt under ``body``, in source order."""
    found: list = []
    for stmt in _flatten_stmts(body):
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Submit):
            found.append(stmt.value)
    return found


def _submit_predicate_operand(submit: ir.Submit) -> ir.Var:
    """The tensor Var a ``Submit``'s first-class ``predicate`` field reads."""
    predicate = submit.predicate
    assert predicate is not None
    return _operand_of(predicate)


def _tuple_result_vars(body, submit: ir.Submit) -> list:
    """Unique ids of the Vars bound from ``submit``'s result tuple.

    The outliner emits ``t = pl.submit(...)`` followed by one
    ``TupleGetItemExpr`` binding per exported output, so the store target the
    gate scope wrote reaches later statements only through one of these.
    """
    stmts = _flatten_stmts(body)
    tuple_vars = {
        stmt.var.unique_id
        for stmt in stmts
        if isinstance(stmt, ir.AssignStmt) and stmt.value is submit and isinstance(stmt.var, ir.Var)
    }
    return [
        stmt.var.unique_id
        for stmt in stmts
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.var, ir.Var)
        and isinstance(stmt.value, ir.TupleGetItemExpr)
        and isinstance(stmt.value.tuple, ir.Var)
        and stmt.value.tuple.unique_id in tuple_vars
    ]


# ``idx`` feeds nothing but the pl.at predicate's index — dead unless DCE counts
# the scope attr as a use. The pl.spmd sibling is
# _SCOPE_PREDICATE_LIVE_INDEX_PROGRAM above.
_AT_PREDICATE_LIVE_INDEX_PROGRAM = """
import pypto.language as pl


@pl.program
class Prog:
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        x: pl.Tensor[[512, 128], pl.FP32],
        out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
        rc: pl.Tensor[[512, 128], pl.INT32],
        i: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[512, 128], pl.FP32]:
        idx = i + 1
        with pl.at(level=pl.Level.CORE_GROUP, predicate=(rc[idx, 0] > 0)):
            v = pl.load(x, [0, 0], [128, 128])
            out = pl.store(v, [0, 0], out)
        return out
"""


def test_dce_keeps_an_at_predicate_operands_producer():
    """A Var used only inside a pl.at scope predicate is a live use, not dead code."""
    program = passes.simplify()(pl.parse_program(_AT_PREDICATE_LIVE_INDEX_PROGRAM))
    main = program.get_function("main")
    assert main is not None
    printed = ir.python_print(main)

    assert "__FREE_VAR" not in printed, f"predicate references a dangling Var:\n{printed}"
    assert "idx: pl.Scalar" in printed, printed


# ---------------------------------------------------------------------------
# The one placement the parser cannot gate: a predicated ``pl.at`` inside an
# ``@pl.function(type=Inline)`` helper. The helper body parses on its own, so
# ``_parse_at_predicate`` sees neither the enclosing scopes nor the caller's
# function type; only ``InlineFunctions`` (pass 1) reveals which call site it
# lands in. The outlining passes re-check it there and report a user-facing
# error, which is why these are ``CHECK`` rather than ``INTERNAL_CHECK``.
# ---------------------------------------------------------------------------

_INLINE_PREDICATE_PROGRAM = """
import pypto.language as pl


@pl.program
class Prog:
    @pl.function(type=pl.FunctionType.Inline)
    def helper(
        self,
        x: pl.Tensor[[512, 128], pl.FP32],
        out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
        rc: pl.Tensor[[512, 128], pl.INT32],
    ) -> pl.Tensor[[512, 128], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, predicate=(rc[0, 0] > 0)):
            out = pl.store(pl.load(x, [0, 0], [128, 128]), [0, 0], out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        x: pl.Tensor[[512, 128], pl.FP32],
        out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
        rc: pl.Tensor[[512, 128], pl.INT32],
    ) -> pl.Tensor[[512, 128], pl.FP32]:
{call_site}
        return out
"""

_OUTLINING_PIPELINE = (
    passes.inline_functions,
    passes.unroll_loops,
    passes.ctrl_flow_transform,
    passes.convert_to_ssa,
    passes.simplify,
    passes.normalize_stmt_structure,
    passes.flatten_call_expr,
    passes.outline_hierarchy_scopes,
    passes.outline_incore_scopes,
    passes.outline_cluster_scopes,
)


def _run_outlining(call_site: str):
    """Parse the inline-helper program with ``call_site`` and outline it.

    Scoped to ``BASIC`` verification on purpose. Under the default ``ROUNDTRIP``
    level the reparse of the printed IR trips the parser's own placement gate
    first — a correct user-facing error too, but a different one, and it would
    hide whether the pass-level check under test fires at all.
    """
    program = pl.parse_program(_INLINE_PREDICATE_PROGRAM.format(call_site=call_site))
    with passes.PassContext([], passes.VerificationLevel.BASIC):
        for factory in _OUTLINING_PIPELINE:
            program = factory()(program)
    return program


@pytest.mark.parametrize(
    "call_site,match",
    [
        (
            "        with pl.cluster():\n            out = self.helper(x, out, rc)",
            r"not supported inside pl\.cluster\(\) / pl\.spmd\(\)",
        ),
        (
            "        for i in pl.spmd(4):\n            out = self.helper(x, out, rc)",
            r"not supported nested inside pl\.spmd\(\)",
        ),
        (
            "        with pl.at(level=pl.Level.CORE_GROUP):\n            out = self.helper(x, out, rc)",
            r"not supported nested inside pl\.spmd\(\)",
        ),
    ],
    ids=["cluster", "spmd", "at"],
)
def test_inline_helper_predicate_in_a_bad_placement_is_rejected(call_site, match):
    """A bad placement smuggled in through an Inline helper fails as a user error.

    ``ValueError`` (from ``CHECK``), not ``InternalError`` — the input is
    ordinary DSL the parser had no way to reject, so blaming the compiler would
    be the wrong classification.
    """
    with pytest.raises(ValueError, match=match):
        _run_outlining(call_site)


def test_inline_helper_predicate_at_top_level_still_lowers():
    """The legal placement must keep working — the gate above is about the caller.

    Pins that the fix is not "reject predicate= in Inline bodies", which would
    drop a surface that lowers correctly.
    """
    program = _run_outlining("        out = self.helper(x, out, rc)")
    main = program.get_function("main")
    assert main is not None
    submits = _submits(main.body)
    assert len(submits) == 1, ir.python_print(main)
    assert submits[0].predicate is not None, ir.python_print(main)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
