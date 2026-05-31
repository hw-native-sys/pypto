# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for NormalizeStmtStructure pass.

This pass normalizes IR structure by:
1. Unwrapping single-child SeqStmts (no redundant nesting)
2. Flattening nested SeqStmts (SeqStmts as child of SeqStmts)

The DSL parser builds function bodies through ``IRBuilder``, whose
``EndFunction``/loop/if scopes already emit ``stmts[0]`` for a single
statement and a flat ``SeqStmts`` for multiple — it never produces a
single-child or nested ``SeqStmts``. So the DSL-authored Before/Expected
tests below only cover the no-op case (the pass leaves an already-flat body
unchanged). The pass's actual rewrites are exercised by the raw ``ir.*``
fixtures ``test_unwraps_single_child_seqstmts`` and
``test_flattens_nested_seqstmts``, which construct the redundant shapes
directly. A final raw-IR test exercises a structurally-invalid mid-body
``YieldStmt`` that the DSL parser likewise cannot emit.
"""

import pypto.language as pl
import pytest
from pypto import DataType, ir, passes


def test_normalize_simple_function():
    """A function body that is a single bare statement is left untouched.

    The DSL parser emits a bare ``ReturnStmt`` (not a single-child
    ``SeqStmts``) for a return-only body, so the pass has nothing to unwrap.
    """

    @pl.program
    class Before:
        @pl.function
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return pl.tensor.adds(x, 1.0)

    After = passes.normalize_stmt_structure()(Before)
    ir.assert_structural_equal(After, Before)


def test_normalize_seqstmts_with_bare_assigns():
    """A flat SeqStmts of bare statements is left untouched.

    A multi-statement function body parses to a flat ``SeqStmts`` (no nesting,
    no single-child wrapping), so the pass leaves it unchanged.
    """

    @pl.program
    class Before:
        @pl.function
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            a = pl.tensor.adds(x, 1.0)
            b = pl.tensor.muls(a, 2.0)
            return b

    After = passes.normalize_stmt_structure()(Before)
    ir.assert_structural_equal(After, Before)


def test_idempotence():
    """Applying normalize twice gives the same result."""

    @pl.program
    class Before:
        @pl.function
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return pl.tensor.adds(x, 1.0)

    After = passes.normalize_stmt_structure()(Before)
    ir.assert_structural_equal(After, Before)

    # Apply pass again and verify idempotence.
    After2 = passes.normalize_stmt_structure()(After)
    ir.assert_structural_equal(After2, Before)


def test_unwraps_single_child_seqstmts():
    """A single-child ``SeqStmts`` function body is unwrapped to the bare stmt.

    The DSL parser never emits a single-child ``SeqStmts`` (``IRBuilder``
    emits ``stmts[0]`` directly), so this pass-input shape is built via raw
    ``ir.*`` constructors.
    """
    span = ir.Span.unknown()
    tensor_ty = ir.TensorType([ir.ConstInt(64, DataType.INT64, span)], DataType.FP32)
    x = ir.Var("x", tensor_ty, span)

    def make_program(body: ir.Stmt) -> ir.Program:
        func = ir.Function("main", [x], [tensor_ty], body, span)
        return ir.Program([func], "test_program", span)

    ret = ir.ReturnStmt([x], span)
    # Before: the function body is a redundant single-child SeqStmts.
    Before = make_program(ir.SeqStmts([ret], span))
    # Expected: the single child is unwrapped to a bare ReturnStmt.
    Expected = make_program(ret)

    # NormalizeStmtStructure repairs NoRedundantBlocks violations, so its input
    # is by definition such a violation — which the structural-property verifier
    # rejects before any pass runs. Disable verification so the repair pass can
    # be exercised on the malformed input it exists to fix.
    with passes.PassContext([], passes.VerificationLevel.NONE):
        After = passes.normalize_stmt_structure()(Before)
    ir.assert_structural_equal(After, Expected)


def test_flattens_nested_seqstmts():
    """A ``SeqStmts`` nested inside another ``SeqStmts`` is flattened to one level.

    Built via raw ``ir.*`` constructors — the DSL parser always emits a flat
    ``SeqStmts`` and never nests one inside another.
    """
    span = ir.Span.unknown()
    tensor_ty = ir.TensorType([ir.ConstInt(64, DataType.INT64, span)], DataType.FP32)
    x = ir.Var("x", tensor_ty, span)
    a = ir.Var("a", tensor_ty, span)
    b = ir.Var("b", tensor_ty, span)

    def make_program(body: ir.Stmt) -> ir.Program:
        func = ir.Function("main", [x], [tensor_ty], body, span)
        return ir.Program([func], "test_program", span)

    assign_a = ir.AssignStmt(
        a,
        ir.Call(ir.get_op("tensor.adds"), [x, ir.ConstFloat(1.0, DataType.FP32, span)], tensor_ty, span),
        span,
    )
    assign_b = ir.AssignStmt(
        b,
        ir.Call(ir.get_op("tensor.muls"), [a, ir.ConstFloat(2.0, DataType.FP32, span)], tensor_ty, span),
        span,
    )
    ret = ir.ReturnStmt([b], span)

    # Before: the body's first child is itself a SeqStmts (one level of nesting).
    Before = make_program(ir.SeqStmts([ir.SeqStmts([assign_a, assign_b], span), ret], span))
    # Expected: the nested SeqStmts is absorbed into a single flat SeqStmts.
    Expected = make_program(ir.SeqStmts([assign_a, assign_b, ret], span))

    # NormalizeStmtStructure repairs NoRedundantBlocks violations, so its input
    # is by definition such a violation — which the structural-property verifier
    # rejects before any pass runs. Disable verification so the repair pass can
    # be exercised on the malformed input it exists to fix.
    with passes.PassContext([], passes.VerificationLevel.NONE):
        After = passes.normalize_stmt_structure()(Before)
    ir.assert_structural_equal(After, Expected)


def test_for_body_flattens_nested_seqstmts():
    """A ForStmt whose body is a nested ``SeqStmts`` gets its body flattened
    while the ``ForStmt`` node (loop var, range, iter_args, return_vars) is
    preserved.

    ``VisitStmt_(ForStmtPtr)`` (normalize_stmt_structure.cpp:127) normalizes
    the body via ``NormalizeBody`` → ``VisitStmt`` → ``VisitStmt_(SeqStmtsPtr)``,
    which absorbs a nested ``SeqStmts`` child (lines 78-82). The loop scaffolding
    is rebuilt via ``MutableCopy`` (lines 138-143), keeping it structurally
    identical. Built via raw ``ir.*`` — the DSL parser never nests SeqStmts.
    """
    span = ir.Span.unknown()
    int_ty = ir.ScalarType(DataType.INT64)
    index_ty = ir.ScalarType(DataType.INDEX)

    a = ir.Var("a", int_ty, span)
    loop_var = ir.Var("i", index_ty, span)
    iter_arg = ir.IterArg("acc", int_ty, a, span)
    rv = ir.Var("result", int_ty, span)

    def make_program(body: ir.Stmt) -> ir.Program:
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
        func = ir.Function("main", [a], [int_ty], func_body, span)
        return ir.Program([func], "test_program", span)

    assign = ir.AssignStmt(
        ir.Var("tmp", int_ty, span),
        ir.Add(iter_arg, loop_var, DataType.INT64, span),
        span,
    )
    yield_stmt = ir.YieldStmt([iter_arg], span)

    # Before: loop body's first child is itself a SeqStmts (one nesting level).
    Before = make_program(ir.SeqStmts([ir.SeqStmts([assign], span), yield_stmt], span))
    # Expected: nested SeqStmts absorbed into a single flat loop body.
    Expected = make_program(ir.SeqStmts([assign, yield_stmt], span))

    with passes.PassContext([], passes.VerificationLevel.NONE):
        After = passes.normalize_stmt_structure()(Before)
    ir.assert_structural_equal(After, Expected)


def test_if_then_and_else_bodies_normalized():
    """An IfStmt with a nested-SeqStmts then-branch and a single-child-SeqStmts
    else-branch gets both branches normalized; the ``IfStmt`` node is preserved.

    ``VisitStmt_(IfStmtPtr)`` (normalize_stmt_structure.cpp:99) normalizes
    ``then_body_`` and ``else_body_`` via ``NormalizeBody``. The then-branch's
    nested ``SeqStmts`` is flattened; the else-branch's single-child ``SeqStmts``
    is unwrapped to the bare child (lines 89-91). Built via raw ``ir.*``.
    """
    span = ir.Span.unknown()
    int_ty = ir.ScalarType(DataType.INT64)

    a = ir.Var("a", int_ty, span)
    rv = ir.Var("result", int_ty, span)
    condition = ir.Gt(a, ir.ConstInt(0, DataType.INT64, span), DataType.BOOL, span)

    assign = ir.AssignStmt(ir.Var("tmp", int_ty, span), a, span)
    then_yield = ir.YieldStmt([a], span)
    else_yield = ir.YieldStmt([ir.ConstInt(0, DataType.INT64, span)], span)

    def make_program(then_body: ir.Stmt, else_body: ir.Stmt) -> ir.Program:
        if_stmt = ir.IfStmt(condition, then_body, else_body, [rv], span)
        func_body = ir.SeqStmts([if_stmt, ir.ReturnStmt([rv], span)], span)
        func = ir.Function("main", [a], [int_ty], func_body, span)
        return ir.Program([func], "test_program", span)

    # Before: then-branch is a nested SeqStmts; else-branch is a single-child SeqStmts.
    Before = make_program(
        ir.SeqStmts([ir.SeqStmts([assign], span), then_yield], span),
        ir.SeqStmts([else_yield], span),
    )
    # Expected: then-branch flattened; else-branch unwrapped to the bare YieldStmt.
    Expected = make_program(
        ir.SeqStmts([assign, then_yield], span),
        else_yield,
    )

    with passes.PassContext([], passes.VerificationLevel.NONE):
        After = passes.normalize_stmt_structure()(Before)
    ir.assert_structural_equal(After, Expected)


def test_while_body_flattens_nested_seqstmts():
    """A WhileStmt whose body is a nested ``SeqStmts`` gets its body flattened
    while the ``WhileStmt`` node (condition, iter_args, return_vars) is preserved.

    ``VisitStmt_(WhileStmtPtr)`` (normalize_stmt_structure.cpp:148) normalizes
    the body via ``NormalizeBody`` and rebuilds the loop via ``MutableCopy``
    (lines 157-160). Built via raw ``ir.*``.
    """
    span = ir.Span.unknown()
    int_ty = ir.ScalarType(DataType.INT64)

    a = ir.Var("a", int_ty, span)
    iter_arg = ir.IterArg("acc", int_ty, a, span)
    rv = ir.Var("result", int_ty, span)
    condition = ir.Gt(iter_arg, ir.ConstInt(0, DataType.INT64, span), DataType.BOOL, span)

    assign = ir.AssignStmt(
        ir.Var("tmp", int_ty, span),
        ir.Add(iter_arg, ir.ConstInt(1, DataType.INT64, span), DataType.INT64, span),
        span,
    )
    yield_stmt = ir.YieldStmt([iter_arg], span)

    def make_program(body: ir.Stmt) -> ir.Program:
        while_stmt = ir.WhileStmt(condition, [iter_arg], body, [rv], span)
        func_body = ir.SeqStmts([while_stmt, ir.ReturnStmt([rv], span)], span)
        func = ir.Function("main", [a], [int_ty], func_body, span)
        return ir.Program([func], "test_program", span)

    # Before: loop body's first child is itself a SeqStmts (one nesting level).
    Before = make_program(ir.SeqStmts([ir.SeqStmts([assign], span), yield_stmt], span))
    # Expected: nested SeqStmts absorbed into a single flat loop body.
    Expected = make_program(ir.SeqStmts([assign, yield_stmt], span))

    with passes.PassContext([], passes.VerificationLevel.NONE):
        After = passes.normalize_stmt_structure()(Before)
    ir.assert_structural_equal(After, Expected)


def test_idempotence_on_malformed_loop_body():
    """Real idempotency: the pass rewrites a malformed (nested-SeqStmts) loop
    body once, then the second application is a no-op.

    Unlike ``test_idempotence`` (which starts from an already-normal body where
    ``Before == Expected``), this case has ``Before != Expected``: the first
    application flattens the nested loop body, and a second application of the
    pass leaves the now-normalized program unchanged.
    """
    span = ir.Span.unknown()
    int_ty = ir.ScalarType(DataType.INT64)
    index_ty = ir.ScalarType(DataType.INDEX)

    a = ir.Var("a", int_ty, span)
    loop_var = ir.Var("i", index_ty, span)
    iter_arg = ir.IterArg("acc", int_ty, a, span)
    rv = ir.Var("result", int_ty, span)

    def make_program(body: ir.Stmt) -> ir.Program:
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
        func = ir.Function("main", [a], [int_ty], func_body, span)
        return ir.Program([func], "test_program", span)

    assign_a = ir.AssignStmt(
        ir.Var("t0", int_ty, span),
        ir.Add(iter_arg, loop_var, DataType.INT64, span),
        span,
    )
    assign_b = ir.AssignStmt(
        ir.Var("t1", int_ty, span),
        ir.Add(iter_arg, iter_arg, DataType.INT64, span),
        span,
    )
    yield_stmt = ir.YieldStmt([iter_arg], span)

    # Before != Expected: loop body has a nested SeqStmts that must be flattened.
    Before = make_program(ir.SeqStmts([ir.SeqStmts([assign_a, assign_b], span), yield_stmt], span))
    Expected = make_program(ir.SeqStmts([assign_a, assign_b, yield_stmt], span))

    with passes.PassContext([], passes.VerificationLevel.NONE):
        After = passes.normalize_stmt_structure()(Before)
        # First application performs the rewrite.
        ir.assert_structural_equal(After, Expected)

        # Second application is a no-op on the now-normalized program.
        After2 = passes.normalize_stmt_structure()(After)
    ir.assert_structural_equal(After2, Expected)


def test_no_redundant_blocks_rejects_mid_body_yield():
    """NoRedundantBlocks rejects a YieldStmt at a non-trailing position of a
    SeqStmts. Function body has no iter_args context, so SSAVerify's
    CheckNoMidBodyYield does not apply — only the structural verifier catches it.

    This input is built via raw ``ir.*`` constructors because the DSL parser
    cannot emit a non-trailing ``YieldStmt`` in a plain function body.
    """
    span = ir.Span.unknown()

    a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
    params: list[ir.Var] = [a]
    return_types: list[ir.Type] = [ir.ScalarType(DataType.INT64)]

    dummy_var = ir.Var("dummy", ir.ScalarType(DataType.INT64), span)
    assign = ir.AssignStmt(dummy_var, a, span)
    mid_yield = ir.YieldStmt([a], span)
    ret = ir.ReturnStmt([a], span)
    func_body = ir.SeqStmts([assign, mid_yield, ret], span)
    func = ir.Function("main", params, return_types, func_body, span)
    program = ir.Program([func], "test_program", span)

    verify_pass = passes.run_verifier()
    with pytest.raises(Exception, match="YieldStmt before the terminating position"):
        verify_pass(program)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
