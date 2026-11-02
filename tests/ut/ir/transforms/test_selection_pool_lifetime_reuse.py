# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compiler-created If selection pools participate in ordinary lifetime reuse."""

import pypto.language as pl
import pytest
from pypto import DataType, ir, passes
from pypto.arith import Analyzer

_SLOT_BYTES = 4 * 8 * 4
_POOL_BYTES = 2 * _SLOT_BYTES
_PLANNERS = [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP]
_ALLOC_OP = ir.get_op("tile.alloc").name


def _storage(program: ir.Program) -> tuple[dict[str, ir.MemRef], dict[str, ir.Call], list[ir.IfStmt]]:
    """Collect storage definitions and If results without relying on generated names."""
    tiles: dict[str, ir.MemRef] = {}
    allocations: dict[str, ir.Call] = {}
    branches: list[ir.IfStmt] = []

    class Collector(ir.IRVisitor):
        def visit_assign_stmt(self, op: ir.AssignStmt) -> None:
            if isinstance(op.var.type, ir.TileType):
                assert op.var.type.memref is not None
                tiles[op.var.name_hint] = op.var.type.memref
            if isinstance(op.value, ir.Call) and op.value.op.name == _ALLOC_OP:
                allocations[op.var.name_hint] = op.value
            super().visit_assign_stmt(op)

        def visit_if_stmt(self, op: ir.IfStmt) -> None:
            branches.append(op)
            super().visit_if_stmt(op)

    Collector().visit_stmt(next(iter(program.functions.values())).body)
    return tiles, allocations, branches


def _constant(expr: ir.Expr) -> int:
    """Require a constant address or slot number."""
    assert isinstance(expr, ir.ConstInt), f"expected a constant, got {expr}"
    return expr.value


def _yielded_slot(body: ir.Stmt | None) -> int:
    """Read a branch's owned trailing yield."""
    while isinstance(body, ir.SeqStmts):
        body = body.stmts[-1]
    assert isinstance(body, ir.YieldStmt) and len(body.value) == 1
    return _constant(body.value[0])


def _assert_reused(program: ir.Program, planner, instruments, *, ordinary: bool = False) -> None:
    """Check pool extent, reuse, and each member's original selection metadata."""
    with passes.PassContext(instruments, memory_planner=planner, enable_buffer_ir=False):
        materialized = passes.materialize_semantic_aliases()(passes.init_mem_ref()(program))
        initial, allocations, branches = _storage(materialized)
        assert len(branches) == 2
        assert not ir.MemRef.same_allocation(initial["a0"], initial["a1"])
        for name in ("a0", "a1"):
            allocation = allocations[initial[name].base_.name_hint]
            assert _constant(allocation.args[1]) == _POOL_BYTES
            assert not allocation.kwargs.get("pinned", False)

        reused = materialized
        if planner == passes.MemoryPlanner.PYPTO:
            reused = passes.memory_reuse()(materialized)
        placed = passes.allocate_memory_addr()(reused)

    tiles, _, branches = _storage(placed)
    starts = []
    for index, branch in enumerate(branches):
        first, second, selected = (tiles[f"{name}{index}"] for name in ("a", "b", "s"))
        start = _constant(first.byte_offset_)
        starts.append(start)
        assert ir.MemRef.same_allocation(first, second)
        assert ir.MemRef.same_allocation(first, selected)
        assert _constant(second.byte_offset_) == start + _SLOT_BYTES
        assert [first.size_, second.size_, selected.size_] == [_SLOT_BYTES] * 3
        assert [first.slot_count_, second.slot_count_, selected.slot_count_] == [2, 2, 2]
        assert first.slot_index_ is not None and _constant(first.slot_index_) == 0
        assert second.slot_index_ is not None and _constant(second.slot_index_) == 1
        assert len(branch.return_vars) == 1
        slot = branch.return_vars[0]
        assert isinstance(slot.type, ir.ScalarType) and slot.type.dtype == DataType.INDEX
        assert selected.slot_index_ is not None
        assert ir.structural_equal(selected.slot_index_, slot)
        assert _yielded_slot(branch.then_body) == 0
        assert _yielded_slot(branch.else_body) == 1
        span = ir.Span.unknown()
        expected = ir.Add(
            ir.ConstInt(start, DataType.INDEX, span),
            ir.Mul(slot, ir.ConstInt(_SLOT_BYTES, DataType.INDEX, span), DataType.INDEX, span),
            DataType.INDEX,
            span,
        )
        difference = Analyzer().simplify(ir.Sub(selected.byte_offset_, expected, DataType.INDEX, span))
        assert _constant(difference) == 0, "reuse lost the selected slot's dynamic byte offset"

    assert starts[0] == starts[1], "nonoverlapping pools must reuse the same physical window"
    if planner == passes.MemoryPlanner.PYPTO:
        assert ir.MemRef.same_allocation(tiles["a0"], tiles["a1"])
    guard = tiles["guard"]
    guard_start = _constant(guard.byte_offset_)
    assert guard.slot_count_ == 1 and guard.slot_index_ is None
    assert guard_start + guard.size_ <= starts[0] or starts[0] + _POOL_BYTES <= guard_start, (
        "the live guard overlaps the pool: address assignment must reserve both slots"
    )
    if ordinary:
        large = tiles["large"]
        assert large.size_ == _POOL_BYTES and _constant(large.byte_offset_) == starts[0]
        assert large.slot_count_ == 1 and large.slot_index_ is None, (
            "ordinary reuse must not inherit the pool's slot metadata"
        )


@pytest.mark.parametrize("planner", _PLANNERS)
def test_sequential_selection_pools_reuse_full_extent(
    planner, pass_verification_instruments, ascend_backend
) -> None:
    """A co-live guard stays outside both slots while two dead pools share storage."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            x: pl.Tensor[[16, 8], pl.FP32],
            flag: pl.Scalar[pl.INT32],
            out: pl.Out[pl.Tensor[[28, 8], pl.FP32]],
        ) -> pl.Tensor[[28, 8], pl.FP32]:
            guard = pl.tile.load(x, [0, 0], [4, 8], target_memory=pl.Mem.Vec)
            a0 = pl.tile.load(x, [0, 0], [4, 8], target_memory=pl.Mem.Vec)
            b0 = pl.tile.load(x, [4, 0], [4, 8], target_memory=pl.Mem.Vec)
            if flag > 0:
                s0 = pl.yield_(a0)
            else:
                s0 = pl.yield_(b0)
            _r0 = pl.tile.store(s0, [0, 0], out)
            _r1 = pl.tile.store(a0, [4, 0], out)
            _r2 = pl.tile.store(b0, [8, 0], out)
            a1 = pl.tile.load(x, [8, 0], [4, 8], target_memory=pl.Mem.Vec)
            b1 = pl.tile.load(x, [12, 0], [4, 8], target_memory=pl.Mem.Vec)
            if flag < 0:
                s1 = pl.yield_(a1)
            else:
                s1 = pl.yield_(b1)
            _r3 = pl.tile.store(s1, [12, 0], out)
            _r4 = pl.tile.store(a1, [16, 0], out)
            _r5 = pl.tile.store(b1, [20, 0], out)
            _r6 = pl.tile.store(guard, [24, 0], out)
            return out

    _assert_reused(Before, planner, pass_verification_instruments)


@pytest.mark.parametrize("planner", _PLANNERS)
def test_selection_pools_reuse_dead_ordinary_allocation(
    planner, pass_verification_instruments, ascend_backend
) -> None:
    """Rebasing onto an ordinary allocation keeps both pools' slot identities."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            x: pl.Tensor[[16, 8], pl.FP32],
            flag: pl.Scalar[pl.INT32],
            out: pl.Out[pl.Tensor[[36, 8], pl.FP32]],
        ) -> pl.Tensor[[36, 8], pl.FP32]:
            guard = pl.tile.load(x, [0, 0], [4, 8], target_memory=pl.Mem.Vec)
            large = pl.tile.load(x, [0, 0], [8, 8], target_memory=pl.Mem.Vec)
            _r_large = pl.tile.store(large, [0, 0], out)
            a0 = pl.tile.load(x, [0, 0], [4, 8], target_memory=pl.Mem.Vec)
            b0 = pl.tile.load(x, [4, 0], [4, 8], target_memory=pl.Mem.Vec)
            if flag > 0:
                s0 = pl.yield_(a0)
            else:
                s0 = pl.yield_(b0)
            _r0 = pl.tile.store(s0, [8, 0], out)
            _r1 = pl.tile.store(a0, [12, 0], out)
            _r2 = pl.tile.store(b0, [16, 0], out)
            a1 = pl.tile.load(x, [8, 0], [4, 8], target_memory=pl.Mem.Vec)
            b1 = pl.tile.load(x, [12, 0], [4, 8], target_memory=pl.Mem.Vec)
            if flag < 0:
                s1 = pl.yield_(a1)
            else:
                s1 = pl.yield_(b1)
            _r3 = pl.tile.store(s1, [20, 0], out)
            _r4 = pl.tile.store(a1, [24, 0], out)
            _r5 = pl.tile.store(b1, [28, 0], out)
            _r6 = pl.tile.store(guard, [32, 0], out)
            return out

    _assert_reused(Before, planner, pass_verification_instruments, ordinary=True)


@pytest.mark.parametrize("planner", _PLANNERS)
def test_ordinary_allocation_reuses_dead_selection_pool(
    planner, pass_verification_instruments, ascend_backend
) -> None:
    """A full-pool-sized ordinary value fits and retains ordinary storage metadata."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            x: pl.Tensor[[16, 8], pl.FP32],
            flag: pl.Scalar[pl.INT32],
            out: pl.Out[pl.Tensor[[36, 8], pl.FP32]],
        ) -> pl.Tensor[[36, 8], pl.FP32]:
            guard = pl.tile.load(x, [0, 0], [4, 8], target_memory=pl.Mem.Vec)
            a0 = pl.tile.load(x, [0, 0], [4, 8], target_memory=pl.Mem.Vec)
            b0 = pl.tile.load(x, [4, 0], [4, 8], target_memory=pl.Mem.Vec)
            if flag > 0:
                s0 = pl.yield_(a0)
            else:
                s0 = pl.yield_(b0)
            _r0 = pl.tile.store(s0, [0, 0], out)
            _r1 = pl.tile.store(a0, [4, 0], out)
            _r2 = pl.tile.store(b0, [8, 0], out)
            large = pl.tile.load(x, [0, 0], [8, 8], target_memory=pl.Mem.Vec)
            _r_large = pl.tile.store(large, [12, 0], out)
            a1 = pl.tile.load(x, [8, 0], [4, 8], target_memory=pl.Mem.Vec)
            b1 = pl.tile.load(x, [12, 0], [4, 8], target_memory=pl.Mem.Vec)
            if flag < 0:
                s1 = pl.yield_(a1)
            else:
                s1 = pl.yield_(b1)
            _r3 = pl.tile.store(s1, [20, 0], out)
            _r4 = pl.tile.store(a1, [24, 0], out)
            _r5 = pl.tile.store(b1, [28, 0], out)
            _r6 = pl.tile.store(guard, [32, 0], out)
            return out

    _assert_reused(Before, planner, pass_verification_instruments, ordinary=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
