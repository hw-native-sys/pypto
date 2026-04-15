# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the PartialUnrollTileLoops pass.

The pass triggers on any ``ForStmt`` carrying ``attrs_["unroll_factor"]``. To
keep these tests focused on the unroll mechanic itself (independent of the
front-end pipeline), they hand-build the input IR with ``ir.*`` constructors.
"""

from typing import cast

import pytest
from pypto import DataType, ir, passes


def _span() -> ir.Span:
    return ir.Span("<test>", 0, 0)


def _const(value: int) -> ir.ConstInt:
    return ir.ConstInt(value, DataType.INDEX, _span())


def _make_program_with_marked_loop(
    *,
    start: int,
    stop: int,
    step: int,
    factor: int,
    body_factory,
) -> tuple[ir.Program, ir.Var]:
    """Build a Program containing one Function whose body is a single ForStmt
    marked with ``unroll_factor=factor``.

    ``body_factory(loop_var)`` returns the body Stmt, given the loop variable.
    """
    span = _span()
    loop_var = ir.Var("i", ir.ScalarType(DataType.INDEX), span)
    body = body_factory(loop_var)
    for_stmt = ir.ForStmt(
        loop_var,
        _const(start),
        _const(stop),
        _const(step),
        [],
        body,
        [],
        span,
        ir.ForKind.Sequential,
        None,
        ir.ChunkPolicy.Guarded,
        {"unroll_factor": factor},
    )
    func_body = ir.SeqStmts([for_stmt, ir.ReturnStmt(span)], span)
    func = ir.Function("main", [], [], func_body, span, ir.FunctionType.Opaque)
    return ir.Program([func], "test", span), loop_var


def _outer_for(program: ir.Program) -> ir.ForStmt:
    func = list(program.functions.values())[0]
    body = cast(ir.SeqStmts, func.body)
    return cast(ir.ForStmt, body.stmts[0])


def _top_level_after(program: ir.Program) -> list[ir.Stmt]:
    func = list(program.functions.values())[0]
    body = cast(ir.SeqStmts, func.body)
    return list(body.stmts)


def _ci(expr) -> int:
    """Cast and unwrap a ConstInt expression's value for assertion."""
    return cast(ir.ConstInt, expr).value


class TestPartialUnrollMechanics:
    """Verify the cloning + outer-stride rewriting logic."""

    def test_clean_divide_produces_replicated_outer_loop(self):
        """trip=8, factor=4 → outer range(0, 8, 4) with body of 4 clones, no remainder."""
        span = _span()
        x = ir.Var("x", ir.ScalarType(DataType.INDEX), span)
        # Body: a single side-effect-free assignment that uses the loop var so we
        # can confirm DeepClone substitutes it across copies.
        Before, _ = _make_program_with_marked_loop(
            start=0,
            stop=8,
            step=1,
            factor=4,
            body_factory=lambda i: ir.AssignStmt(x, i, span),
        )

        After = passes.partial_unroll_tile_loops()(Before)
        stmts = _top_level_after(After)

        # No remainder loop expected — single outer ForStmt remains.
        assert len(stmts) == 2  # outer for + return
        outer = cast(ir.ForStmt, stmts[0])

        # Outer loop: range(0, 8, 4)
        assert _ci(outer.start) == 0
        assert _ci(outer.stop) == 8
        assert _ci(outer.step) == 4

        # Marker attr present, factor attr removed.
        attrs = dict(outer.attrs)
        assert attrs.get("unroll_replicated") == 4
        assert "unroll_factor" not in attrs

        # Body has F copies.
        body_stmts = list(cast(ir.SeqStmts, outer.body).stmts)
        assert len(body_stmts) == 4

    def test_with_remainder_appends_remainder_loop(self):
        """trip=10, factor=4 → main range(0,8,4) with 4 clones + remainder range(8,10,1)."""
        span = _span()
        x = ir.Var("x", ir.ScalarType(DataType.INDEX), span)
        Before, _ = _make_program_with_marked_loop(
            start=0,
            stop=10,
            step=1,
            factor=4,
            body_factory=lambda i: ir.AssignStmt(x, i, span),
        )

        After = passes.partial_unroll_tile_loops()(Before)
        stmts = _top_level_after(After)
        # Expect: main loop, remainder loop, return
        assert len(stmts) == 3

        main = cast(ir.ForStmt, stmts[0])
        rem = cast(ir.ForStmt, stmts[1])

        # Main covers iterations [0, 8) with stride 4.
        assert _ci(main.start) == 0
        assert _ci(main.stop) == 8
        assert _ci(main.step) == 4
        assert dict(main.attrs).get("unroll_replicated") == 4

        # Remainder covers [8, 10) with original stride 1, no marker.
        assert _ci(rem.start) == 8
        assert _ci(rem.stop) == 10
        assert _ci(rem.step) == 1
        assert "unroll_replicated" not in dict(rem.attrs)
        assert "unroll_factor" not in dict(rem.attrs)

    def test_factor_one_is_noop(self):
        """unroll_factor=1 leaves the loop intact (modulo attr cleanup)."""
        span = _span()
        x = ir.Var("x", ir.ScalarType(DataType.INDEX), span)
        Before, _ = _make_program_with_marked_loop(
            start=0,
            stop=8,
            step=1,
            factor=1,
            body_factory=lambda i: ir.AssignStmt(x, i, span),
        )

        After = passes.partial_unroll_tile_loops()(Before)
        outer = _outer_for(After)
        # Range untouched.
        assert _ci(outer.start) == 0
        assert _ci(outer.stop) == 8
        assert _ci(outer.step) == 1
        # Marker NOT added; factor attr removed.
        attrs = dict(outer.attrs)
        assert "unroll_factor" not in attrs
        assert "unroll_replicated" not in attrs

    def test_factor_equals_trip_count(self):
        """factor=4, trip=4 → single outer iteration containing 4 clones, no remainder."""
        span = _span()
        x = ir.Var("x", ir.ScalarType(DataType.INDEX), span)
        Before, _ = _make_program_with_marked_loop(
            start=0,
            stop=4,
            step=1,
            factor=4,
            body_factory=lambda i: ir.AssignStmt(x, i, span),
        )

        After = passes.partial_unroll_tile_loops()(Before)
        stmts = _top_level_after(After)
        assert len(stmts) == 2  # only main loop + return
        outer = cast(ir.ForStmt, stmts[0])

        # main_iters * factor * step = 1 * 4 * 1 = 4 → outer stop is 4.
        assert _ci(outer.start) == 0
        assert _ci(outer.stop) == 4
        assert _ci(outer.step) == 4
        assert len(cast(ir.SeqStmts, outer.body).stmts) == 4

    def test_iter_args_rejected_with_clear_message(self):
        """Loops with iter_args must be rejected — partial unroll cannot handle loop-carried state."""
        span = _span()
        loop_var = ir.Var("i", ir.ScalarType(DataType.INDEX), span)
        init = ir.ConstInt(0, DataType.INT64, span)
        iter_arg = ir.IterArg("acc", ir.ScalarType(DataType.INT64), init, span)
        rv = ir.Var("acc_rv", ir.ScalarType(DataType.INT64), span)
        # YieldStmt directly as the body keeps NoRedundantBlocks happy.
        body = ir.YieldStmt([init], span)
        for_stmt = ir.ForStmt(
            loop_var,
            _const(0),
            _const(8),
            _const(1),
            [iter_arg],
            body,
            [rv],
            span,
            ir.ForKind.Sequential,
            None,
            ir.ChunkPolicy.Guarded,
            {"unroll_factor": 4},
        )
        func_body = ir.SeqStmts([for_stmt, ir.ReturnStmt(span)], span)
        func = ir.Function("main", [], [], func_body, span, ir.FunctionType.Opaque)
        Before = ir.Program([func], "test", span)

        # Disable the structural pre-verifier — the test input is intentionally
        # tiny; we just want to confirm the iter_args precondition is enforced.
        with passes.PassContext([], passes.VerificationLevel.NONE):
            with pytest.raises(ValueError, match="iter_args"):
                passes.partial_unroll_tile_loops()(Before)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
