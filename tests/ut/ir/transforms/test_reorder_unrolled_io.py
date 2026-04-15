# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the ReorderUnrolledIO pass.

The pass walks unroll-replicated regions (``ForStmt`` carrying
``attrs_["unroll_replicated"]``) and clusters tile.load to the top, tile.store
to the bottom of each marked SeqStmts body — subject to the dependency graph.
Tests build minimal IR by hand to keep the focus on the reorder mechanic.
"""

from collections.abc import Sequence
from typing import cast

import pytest
from pypto import DataType, ir, passes
from pypto.ir.op import tile


def _span() -> ir.Span:
    return ir.Span("<test>", 0, 0)


def _const(value: int, dtype: DataType = DataType.INDEX) -> ir.ConstInt:
    return ir.ConstInt(value, dtype, _span())


def _mk_tile_var(name: str) -> ir.Var:
    span = _span()
    dim = _const(64, DataType.INT64)
    return ir.Var(
        name,
        ir.TileType([dim, dim], DataType.FP32, memory_space=ir.MemorySpace.Vec),
        span,
    )


def _mk_tensor_var(name: str) -> ir.Var:
    return ir.Var(name, ir.TensorType([64, 64], DataType.FP32), _span())


def _wrap_program(body_stmts: Sequence[ir.Stmt], unroll_replicated: int | None = 4) -> ir.Program:
    """Build a Program: function whose body is one ForStmt over ``body_stmts``,
    optionally tagged with ``unroll_replicated``."""
    span = _span()
    loop_var = ir.Var("i", ir.ScalarType(DataType.INDEX), span)
    seq = ir.SeqStmts(list(body_stmts), span)
    attrs: dict[str, object] = {}
    if unroll_replicated is not None:
        attrs["unroll_replicated"] = unroll_replicated
    for_stmt = ir.ForStmt(
        loop_var,
        _const(0),
        _const(2),
        _const(1),
        [],
        seq,
        [],
        span,
        ir.ForKind.Sequential,
        None,
        ir.ChunkPolicy.Guarded,
        attrs,
    )
    func_body = ir.SeqStmts([for_stmt, ir.ReturnStmt(span)], span)
    func = ir.Function("main", [], [], func_body, span, ir.FunctionType.Opaque)
    return ir.Program([func], "test", span)


def _seq_after(program: ir.Program) -> list[ir.Stmt]:
    """Top-level statements of the inner SeqStmts after the pass."""
    func = list(program.functions.values())[0]
    func_body = cast(ir.SeqStmts, func.body)
    outer = cast(ir.ForStmt, func_body.stmts[0])
    return list(cast(ir.SeqStmts, outer.body).stmts)


def _is_load(stmt: ir.Stmt) -> bool:
    if not isinstance(stmt, ir.AssignStmt):
        return False
    call = stmt.value
    return isinstance(call, ir.Call) and call.op.name == "tile.load"


def _is_store(stmt: ir.Stmt) -> bool:
    if not isinstance(stmt, ir.AssignStmt):
        return False
    call = stmt.value
    return isinstance(call, ir.Call) and call.op.name == "tile.store"


def _run_pass(program: ir.Program) -> ir.Program:
    """Run ReorderUnrolledIO with structural verification disabled — the test
    inputs are intentionally minimal and don't satisfy the full set of
    structural prerequisites the pipeline normally enforces."""
    with passes.PassContext([], passes.VerificationLevel.NONE):
        return passes.reorder_unrolled_io()(program)


class TestReorderUnrolledIO:
    """Verify the priority-aware topological reorder."""

    def test_symmetric_pingpong_layout(self):
        """[load_0, compute_0, store_0, load_1, compute_1, store_1] →
        [load_0, load_1, compute_0, compute_1, store_0, store_1]."""
        span = _span()
        in_a, in_b, out = _mk_tensor_var("in_a"), _mk_tensor_var("in_b"), _mk_tensor_var("out")
        ta0, tb0, tc0 = _mk_tile_var("ta0"), _mk_tile_var("tb0"), _mk_tile_var("tc0")
        ta1, tb1, tc1 = _mk_tile_var("ta1"), _mk_tile_var("tb1"), _mk_tile_var("tc1")
        sr0, sr1 = _mk_tensor_var("sr0"), _mk_tensor_var("sr1")
        body = [
            ir.AssignStmt(ta0, tile.load(in_a, offsets=[0, 0], shapes=[64, 64]), span),
            ir.AssignStmt(tc0, tile.add(ta0, ta0), span),
            ir.AssignStmt(sr0, tile.store(tc0, offsets=[0, 0], output_tensor=out), span),
            ir.AssignStmt(tb0, tile.load(in_b, offsets=[0, 0], shapes=[64, 64]), span),
            ir.AssignStmt(ta1, tile.load(in_a, offsets=[64, 0], shapes=[64, 64]), span),
            ir.AssignStmt(tc1, tile.add(ta1, ta1), span),
            ir.AssignStmt(sr1, tile.store(tc1, offsets=[64, 0], output_tensor=out), span),
            ir.AssignStmt(tb1, tile.load(in_b, offsets=[64, 0], shapes=[64, 64]), span),
        ]
        Before = _wrap_program(body, unroll_replicated=2)
        After = _run_pass(Before)
        out_stmts = _seq_after(After)

        # All loads first, then computes, then stores — order within each band preserved.
        n_load = sum(1 for s in out_stmts if _is_load(s))
        n_store = sum(1 for s in out_stmts if _is_store(s))
        assert n_load == 4
        assert n_store == 2

        load_idx = [i for i, s in enumerate(out_stmts) if _is_load(s)]
        store_idx = [i for i, s in enumerate(out_stmts) if _is_store(s)]
        compute_idx = [i for i, s in enumerate(out_stmts) if not _is_load(s) and not _is_store(s)]
        assert max(load_idx) < min(compute_idx) < min(store_idx) <= max(store_idx)

    def test_load_blocked_by_compute_stays_put(self):
        """A load that depends on a compute stmt cannot float past it."""
        span = _span()
        in_a, out = _mk_tensor_var("in_a"), _mk_tensor_var("out")
        ta, tc, sr = _mk_tile_var("ta"), _mk_tile_var("tc"), _mk_tensor_var("sr")
        # offset_var is computed mid-body; the second load's offset uses it,
        # creating a load→non-load dependency that must be preserved.
        offset_var = ir.Var("off", ir.ScalarType(DataType.INDEX), span)
        ta2 = _mk_tile_var("ta2")
        body = [
            ir.AssignStmt(ta, tile.load(in_a, offsets=[0, 0], shapes=[64, 64]), span),
            ir.AssignStmt(offset_var, _const(64), span),
            ir.AssignStmt(ta2, tile.load(in_a, offsets=[offset_var, _const(0)], shapes=[64, 64]), span),
            ir.AssignStmt(tc, tile.add(ta, ta2), span),
            ir.AssignStmt(sr, tile.store(tc, offsets=[0, 0], output_tensor=out), span),
        ]
        Before = _wrap_program(body)
        After = _run_pass(Before)
        out_stmts = _seq_after(After)

        # Find indices of ta-defining load and ta2-defining load.
        ta_idx = next(i for i, s in enumerate(out_stmts) if isinstance(s, ir.AssignStmt) and s.var is ta)
        offset_idx = next(
            i for i, s in enumerate(out_stmts) if isinstance(s, ir.AssignStmt) and s.var is offset_var
        )
        ta2_idx = next(i for i, s in enumerate(out_stmts) if isinstance(s, ir.AssignStmt) and s.var is ta2)

        # The independent first load lifts to the very top.
        assert ta_idx == 0
        # The dependent second load stays after the offset assignment.
        assert offset_idx < ta2_idx

    def test_skip_unmarked_for(self):
        """A ForStmt without ``unroll_replicated`` is left untouched."""
        span = _span()
        in_a, out = _mk_tensor_var("in_a"), _mk_tensor_var("out")
        ta, tc, sr = _mk_tile_var("ta"), _mk_tile_var("tc"), _mk_tensor_var("sr")
        body = [
            ir.AssignStmt(ta, tile.load(in_a, offsets=[0, 0], shapes=[64, 64]), span),
            ir.AssignStmt(tc, tile.add(ta, ta), span),
            ir.AssignStmt(sr, tile.store(tc, offsets=[0, 0], output_tensor=out), span),
        ]
        Before = _wrap_program(body, unroll_replicated=None)
        After = _run_pass(Before)
        # IR identity is preserved when no marker is present.
        assert After is Before

    def test_no_io_ops_is_noop(self):
        """A marked region with neither loads nor stores is unchanged."""
        span = _span()
        ta = _mk_tile_var("ta")
        tb = _mk_tile_var("tb")
        body = [
            ir.AssignStmt(ta, tile.add(ta, ta), span),
            ir.AssignStmt(tb, tile.add(tb, tb), span),
        ]
        Before = _wrap_program(body)
        After = _run_pass(Before)
        # The marked outer ForStmt's body is still a 2-element SeqStmts in the same order.
        out_stmts = _seq_after(After)
        assert len(out_stmts) == 2
        assert isinstance(out_stmts[0], ir.AssignStmt) and out_stmts[0].var is ta
        assert isinstance(out_stmts[1], ir.AssignStmt) and out_stmts[1].var is tb

    def test_relative_order_preserved_among_independent_loads(self):
        """3 independent loads keep their original relative order after lifting."""
        span = _span()
        in_a = _mk_tensor_var("in_a")
        ta0, ta1, ta2 = _mk_tile_var("ta0"), _mk_tile_var("ta1"), _mk_tile_var("ta2")
        out = _mk_tensor_var("out")
        sr = _mk_tensor_var("sr")
        tc = _mk_tile_var("tc")
        body = [
            ir.AssignStmt(ta0, tile.load(in_a, offsets=[0, 0], shapes=[64, 64]), span),
            ir.AssignStmt(tc, tile.add(ta0, ta0), span),
            ir.AssignStmt(ta1, tile.load(in_a, offsets=[64, 0], shapes=[64, 64]), span),
            ir.AssignStmt(ta2, tile.load(in_a, offsets=[128, 0], shapes=[64, 64]), span),
            ir.AssignStmt(sr, tile.store(tc, offsets=[0, 0], output_tensor=out), span),
        ]
        Before = _wrap_program(body)
        After = _run_pass(Before)
        out_stmts = _seq_after(After)
        # Loads cluster at the top in their original relative order (ta0, ta1, ta2).
        load_vars = [s.var for s in out_stmts if isinstance(s, ir.AssignStmt) and _is_load(s)]
        assert load_vars == [ta0, ta1, ta2]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
