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
from pypto import DataType, codegen, ir, passes
from pypto.arith import Analyzer
from pypto.backend import pto_backend
from pypto.backend._ptoas_locate import find_ptoas_binary

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


def _conditional_reset_program(
    *, reset_in_else: bool = False, old_use: str = "none", protection: str = "none"
) -> ir.Program:
    """Vary the old payload's observation without changing the selected result."""
    old_store = "old_out = pl.tile.store(current, [4, 0], out)"
    before = old_store if old_use == "before" else ""
    tail = old_store if old_use == "tail" else ""
    producer = (
        "pl.tile.recip(current)" if old_use == "operand" else "pl.tile.full([4, 8], dtype=pl.FP32, value=0.0)"
    )
    current_annotation = (
        ': pl.Tile[[4, 8], pl.FP32, pl.MemRef("current_storage", slots=2)[0], pl.Mem.Vec]'
        if protection == "pinned"
        else ""
    )
    load_attrs = ', attrs={"pipeline_membership": "0:0"}' if protection == "pipeline" else ""
    if protection == "pipeline":
        producer = 'pl.tile.full([4, 8], dtype=pl.FP32, value=0.0, attrs={"pipeline_membership": "0:1"})'
    workspace = ""
    if protection == "workspace":
        workspace = """source = pl.tile.full([4, 8], dtype=pl.FP32, value=1.0)
        mask = pl.tile.cmps(source, 0.0, cmp_type=4)
        workspace_result = pl.tile.sels(mask, source, current, -1.0)
        workspace_out = pl.tile.store(workspace_result, [4, 0], out)"""
    reset = f"""{before}
            reset = {producer}
            {tail}
            selected = pl.yield_(reset)"""
    keep = "selected = pl.yield_(current)"
    if old_use == "sibling":
        keep = f"{old_store}\n            {keep}"
    then, otherwise = (keep, reset) if reset_in_else else (reset, keep)
    early_alias = "alias = current" if old_use == "alias_after" else ""
    after = {
        "after": old_store,
        "alias_after": "old_out = pl.tile.store(alias, [4, 0], out)",
        "late_alias": "alias = current\n        old_out = pl.tile.store(alias, [4, 0], out)",
    }.get(old_use, "")
    return pl.parse_program(f"""
@pl.program
class ConditionalReset:
    @pl.function(type=pl.FunctionType.AIV)
    def main(self, pool: pl.Tensor[[4, 8], pl.FP32], flag: pl.Scalar[pl.INT32],
             out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]) -> pl.Tensor[[8, 8], pl.FP32]:
        current{current_annotation} = pl.tile.load(pool, [0, 0], [4, 8], target_memory=pl.Mem.Vec{load_attrs})
        {workspace}
        {early_alias}
        if flag != 0:
            {then}
        else:
            {otherwise}
        {after}
        result = pl.tile.store(selected, [0, 0], out)
        return result
""")


@pytest.mark.parametrize("planner", [*_PLANNERS, passes.MemoryPlanner.PTOAS])
@pytest.mark.parametrize("reset_in_else", [False, True])
@pytest.mark.parametrize(
    "old_use", ["none", "before", "sibling", "tail", "after", "alias_after", "late_alias", "operand"]
)
def test_conditional_selection_slot_liveness(
    planner, reset_in_else, old_use, pass_verification_instruments, ascend_backend
) -> None:
    """Only observations that cannot follow the overwrite allow one shared slot."""
    before = _conditional_reset_program(reset_in_else=reset_in_else, old_use=old_use)
    with passes.PassContext(pass_verification_instruments, memory_planner=planner, enable_buffer_ir=False):
        materialized = passes.materialize_semantic_aliases()(passes.init_mem_ref()(before))
        tiles, allocations, branches = _storage(materialized)
        current, reset, selected = (tiles[name] for name in ("current", "reset", "selected"))
        shared = old_use in {"none", "before", "sibling"}
        assert ir.MemRef.same_allocation(current, reset)
        assert current.slot_count_ == reset.slot_count_ == selected.slot_count_ == (1 if shared else 2)
        assert _constant(current.byte_offset_) == 0
        assert _constant(reset.byte_offset_) == (0 if shared else _SLOT_BYTES)
        assert _constant(allocations[current.base_.name_hint].args[1]) == (
            _SLOT_BYTES if shared else _POOL_BYTES
        )
        if shared:
            assert all(ref.slot_index_ is None for ref in (current, reset, selected))
            assert _constant(selected.byte_offset_) == 0
            assert _yielded_slot(branches[0].then_body) == _yielded_slot(branches[0].else_body) == 0
            body = branches[0].else_body if reset_in_else else branches[0].then_body
            assert isinstance(body, ir.SeqStmts)
            reset_position = next(
                i
                for i, stmt in enumerate(body.stmts)
                if isinstance(stmt, ir.AssignStmt) and stmt.var.name_hint == "reset"
            )
            barrier = body.stmts[reset_position - 1]
            assert reset_position > 0 and isinstance(barrier, ir.EvalStmt)
            assert isinstance(barrier.expr, ir.Call)
            assert barrier.expr.op.name == ir.get_op("system.bar_all").name
        # Once the choice has become scalar control flow, rerunning semantic
        # materialization must preserve both capacity and synchronization.
        ir.assert_structural_equal(passes.materialize_semantic_aliases()(materialized), materialized)
        if planner == passes.MemoryPlanner.PYPTO:
            materialized = passes.memory_reuse()(materialized)
        if planner != passes.MemoryPlanner.PTOAS:
            passes.allocate_memory_addr()(materialized)


@pytest.mark.parametrize("protection", ["pinned", "pipeline", "workspace"])
@pytest.mark.parametrize("planner", [*_PLANNERS, passes.MemoryPlanner.PTOAS])
def test_conditional_selection_preserves_storage_constraints(
    protection, planner, pass_verification_instruments, ascend_backend
) -> None:
    """A branch-dead payload does not override independent storage constraints."""
    before = _conditional_reset_program(protection=protection)
    with passes.PassContext(pass_verification_instruments, memory_planner=planner, enable_buffer_ir=False):
        after = passes.materialize_semantic_aliases()(passes.init_mem_ref()(before))
    tiles, _, _ = _storage(after)
    current, reset = tiles["current"], tiles["reset"]
    assert not ir.MemRef.same_allocation(current, reset) or (
        _constant(current.byte_offset_) + current.size_ <= _constant(reset.byte_offset_)
        or _constant(reset.byte_offset_) + reset.size_ <= _constant(current.byte_offset_)
    )


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("planner", [*_PLANNERS, passes.MemoryPlanner.PTOAS])
@pytest.mark.parametrize("reset_in_else", [False, True])
def test_conditional_selection_single_slot_ptoas(
    planner, reset_in_else, pass_verification_instruments, ascend_backend, tmp_path
) -> None:
    """The shared static descriptor and both synchronization points lower to PTOAS."""
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    before = _conditional_reset_program(reset_in_else=reset_in_else)
    with passes.PassContext(pass_verification_instruments, memory_planner=planner, enable_buffer_ir=False):
        after = passes.materialize_semantic_aliases()(passes.init_mem_ref()(before))
        if planner == passes.MemoryPlanner.PYPTO:
            after = passes.memory_reuse()(after)
        if planner != passes.MemoryPlanner.PTOAS:
            after = passes.allocate_memory_addr()(after)
        modules = codegen.PTOCodegen().generate(after, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS)
        mlir = modules if isinstance(modules, str) else "\n".join(modules.values())
        assert "pto.alloc_multi_tile" not in mlir
        cpp = pto_backend._compile_pto_module(mlir, "main", str(tmp_path), memory_planner=planner)
        assert "main" in cpp


@pytest.mark.parametrize("planner", [*_PLANNERS, passes.MemoryPlanner.PTOAS])
def test_conditional_selection_preserves_old_value_forwarded_by_another_if(
    planner, pass_verification_instruments, ascend_backend
) -> None:
    """A later selection may demand the old value without a direct payload read."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def main(
            self,
            pool: pl.Tensor[[4, 8], pl.FP32],
            flag: pl.Scalar[pl.INT32],
            other_flag: pl.Scalar[pl.INT32],
            out: pl.Out[pl.Tensor[[4, 8], pl.FP32]],
        ) -> pl.Tensor[[4, 8], pl.FP32]:
            current = pl.tile.load(pool, [0, 0], [4, 8], target_memory=pl.Mem.Vec)
            if flag != 0:
                reset = pl.tile.full([4, 8], dtype=pl.FP32, value=0.0)
                first = pl.yield_(reset)
            else:
                first = pl.yield_(current)
            if other_flag != 0:
                selected = pl.yield_(current)
            else:
                selected = pl.yield_(first)
            return pl.tile.store(selected, [0, 0], out)

    with passes.PassContext(pass_verification_instruments, memory_planner=planner, enable_buffer_ir=False):
        after = passes.materialize_semantic_aliases()(passes.init_mem_ref()(Before))
    tiles, allocations, branches = _storage(after)
    current, reset = tiles["current"], tiles["reset"]
    assert len(branches) == 2
    assert current.slot_count_ == reset.slot_count_ == 2
    assert _constant(allocations[current.base_.name_hint].args[1]) == _POOL_BYTES
    assert _constant(current.byte_offset_) != _constant(reset.byte_offset_)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
