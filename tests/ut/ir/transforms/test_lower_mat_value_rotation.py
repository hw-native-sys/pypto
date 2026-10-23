# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Check scalarized Mat selectors by interpreting the selected storage windows."""

import operator
from typing import Any

import pypto.language as pl
import pytest
from pypto import ir, passes

_PANEL = "pl.Tile[[64, 64], pl.FP16, pl.Mem.Mat]"


def _program(*, start=0, step=1, skip=False, preserve_seed=False, declared_seed=False, repeat=False):
    guard = f"block + {step} < stop"
    if skip:
        guard += f" and ((block - {start}) // {step}) % 3 != 1"
    saved = f"saved: {_PANEL} = seed" if preserve_seed else ""
    observe = (
        "saved_read: pl.Tile[[64, 64], pl.FP16, pl.Mem.Left] = pl.tile.move(saved, target_memory=pl.Mem.Left)"
        if preserve_seed
        else ""
    )
    seed_type = (
        'pl.Tile[[64, 64], pl.FP16, pl.MemRef("declared", slots=2)[0], pl.Mem.Mat]'
        if declared_seed
        else _PANEL
    )
    source = f"""
@pl.program
class Rotation:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[2048, 64], pl.FP16],
               stop: pl.Scalar[pl.INDEX]) -> pl.Tensor[[2048, 64], pl.FP16]:
        seed: {seed_type} = pl.tile.load(a, [{start} * 64, 0], [64, 64], target_memory=pl.Mem.Mat)
        {saved}
        for block, (current,) in pl.range({start}, stop, {step}, init_values=(seed,)):
            if {guard}:
                following: {_PANEL} = pl.tile.load(
                    a, [(block + {step}) * 64, 0], [64, 64], target_memory=pl.Mem.Mat)
            else:
                following: {_PANEL} = current
            consumed: pl.Tile[[64, 64], pl.FP16, pl.Mem.Left] = pl.tile.move(
                current, target_memory=pl.Mem.Left)
            final = pl.yield_(following)
        final_read: pl.Tile[[64, 64], pl.FP16, pl.Mem.Left] = pl.tile.move(
            final, target_memory=pl.Mem.Left)
        {observe}
        return a
"""
    if repeat:
        start_index = source.index("        for block,")
        end_index = source.index("        return a", start_index)
        source = (
            source[:start_index]
            + "        for repetition in pl.range(2):\n"
            + "".join("    " + line + "\n" for line in source[start_index:end_index].splitlines())
            + source[end_index:]
        )
    return pl.parse_program(source)


def _lower(program):
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO):
        before = passes.convert_to_ssa()(program)
        return before, passes.lower_mat_value_rotation()(before)


class _StorageInterpreter:
    """Only scalar/control flow plus load/create/move; no numerical kernel emulation.

    A load stores its GM block number in the addressed physical slot. A Mat->Left
    read retrieves that number. Inverting a selector or overwriting the current
    slot before consuming it therefore produces a different observable sequence.
    """

    _binary = {
        ir.Add: operator.add,
        ir.Sub: operator.sub,
        ir.Mul: operator.mul,
        ir.FloorDiv: operator.floordiv,
        ir.FloorMod: operator.mod,
        ir.Lt: operator.lt,
        ir.Ne: operator.ne,
        ir.Eq: operator.eq,
        ir.And: operator.and_,
    }

    def __init__(self, program, stop):
        self.function = next(iter(program.functions.values()))
        self.values: dict[int, Any] = {
            self.function.params[0].unique_id: "a",
            self.function.params[1].unique_id: stop,
        }
        self.storage: dict[tuple[int, int], int] = {}
        self.slot_counts: set[int] = set()
        self.observed: list[int] = []
        self.writes: list[tuple[tuple[int, int], int]] = []

    def evaluate(self, expr: ir.Expr) -> Any:
        if isinstance(expr, ir.ConstInt):
            return expr.value
        if isinstance(expr, ir.ConstBool):
            return expr.value
        if isinstance(expr, (ir.Var, ir.IterArg)):
            return self.values[expr.unique_id]
        if isinstance(expr, ir.MakeTuple):
            return [self.evaluate(item) for item in expr.elements]
        for kind, operation in self._binary.items():
            if isinstance(expr, kind):
                return operation(self.evaluate(expr.left), self.evaluate(expr.right))
        raise AssertionError(f"Unexpected scalar expression: {expr}")

    def slot(self, var: ir.Var) -> tuple[int, int]:
        assert isinstance(var.type, ir.TileType)
        memref = var.type.memref
        assert memref is not None and memref.slot_index_ is not None, var
        self.slot_counts.add(memref.slot_count_)
        return memref.base_.unique_id, self.evaluate(memref.slot_index_)

    def execute(self, stmt: ir.Stmt) -> list[Any] | None:
        if isinstance(stmt, ir.SeqStmts):
            for child in stmt.stmts:
                yielded = self.execute(child)
                if yielded is not None:
                    return yielded
        elif isinstance(stmt, ir.AssignStmt):
            if not isinstance(stmt.value, ir.Call):
                self.values[stmt.var.unique_id] = self.evaluate(stmt.value)
                return None
            self.execute_call(stmt)
        elif isinstance(stmt, ir.IfStmt):
            body = stmt.then_body if self.evaluate(stmt.condition) else stmt.else_body
            assert body is not None
            yielded = self.execute(body)
            assert yielded is not None
            for result, value in zip(stmt.return_vars, yielded, strict=True):
                self.values[result.unique_id] = value
        elif isinstance(stmt, ir.ForStmt):
            assert all(isinstance(arg.type, ir.ScalarType) for arg in stmt.iter_args)
            state = [self.evaluate(argument.initValue) for argument in stmt.iter_args]
            for index in range(self.evaluate(stmt.start), self.evaluate(stmt.stop), self.evaluate(stmt.step)):
                self.values[stmt.loop_var.unique_id] = index
                for argument, value in zip(stmt.iter_args, state, strict=True):
                    self.values[argument.unique_id] = value
                state = self.execute(stmt.body) or []
            for result, value in zip(stmt.return_vars, state, strict=True):
                self.values[result.unique_id] = value
        elif isinstance(stmt, ir.YieldStmt):
            return [self.evaluate(value) for value in stmt.value]
        elif not isinstance(stmt, ir.ReturnStmt):
            raise AssertionError(f"Unexpected statement: {stmt}")
        return None

    def execute_call(self, stmt: ir.AssignStmt) -> None:
        call = stmt.value
        assert isinstance(call, ir.Call)
        if call.op.name == ir.get_op("tile.load").name:
            key = self.slot(stmt.var)
            block = self.evaluate(call.args[1])[0] // 64
            self.storage[key] = block
            self.writes.append((key, block))
            self.values[stmt.var.unique_id] = key
        elif call.op.name == ir.get_op("tile.create").name:
            self.values[stmt.var.unique_id] = self.slot(stmt.var)
        elif call.op.name == ir.get_op("tile.move").name:
            key = self.evaluate(call.args[0])
            self.observed.append(self.storage[key])
            self.values[stmt.var.unique_id] = self.storage[key]
        else:
            raise AssertionError(f"Unexpected call: {call.op.name}")

    def run(self):
        self.execute(self.function.body)
        return self.observed


@pytest.mark.parametrize("start,step", [(0, 1), (2, 3)])
@pytest.mark.parametrize("trips", [0, 1, 3, 4, 7, 8])
@pytest.mark.parametrize("skip", [False, True])
@pytest.mark.parametrize("preserve_seed", [False, True])
def test_rotating_slots_preserve_values(start, step, trips, skip, preserve_seed):
    """Check data meaning, arbitrary conditional retains, odd/even and zero trips."""
    stop = start + step * trips
    _, lowered = _lower(_program(start=start, step=step, skip=skip, preserve_seed=preserve_seed))
    current = start
    expected = []
    for iteration, block in enumerate(range(start, stop, step)):
        expected.append(current)
        if block + step < stop and (not skip or iteration % 3 != 1):
            current = block + step
    expected.append(current)  # The post-loop consumer uses the final selected bank.
    if preserve_seed:
        expected.append(start)
    interpreter = _StorageInterpreter(lowered, stop)
    assert interpreter.run() == expected, lowered.as_python()
    assert interpreter.slot_counts == {3 if preserve_seed else 2}
    # Generated declarations and scalar binders survive the printer/parser and
    # reapplying the pass cannot create another bank family.
    ir.assert_structural_equal(pl.parse_program(lowered.as_python()), lowered)
    ir.assert_structural_equal(passes.lower_mat_value_rotation()(lowered), lowered)


def test_init_memref_resolves_slot_selector_to_rebuilt_scalar_carry():
    """Resolved offset/index must follow InitMemRef's new IterArg identity."""
    program = pl.parse_program("""
@pl.program
class SelectedSlot:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[64, 64], pl.FP16]) -> pl.Tensor[[64, 64], pl.FP16]:
        initial_slot: pl.Scalar[pl.INDEX] = 0
        alias_slot: pl.Scalar[pl.INDEX] = initial_slot
        for i, (slot,) in pl.range(0, 3, init_values=(alias_slot,)):
            value: pl.Tile[[64, 64], pl.FP16, pl.MemRef("selected", slots=2)[slot], pl.Mem.Mat] = (
                pl.tile.load(a, [0, 0], [64, 64], target_memory=pl.Mem.Mat)
            )
            final = pl.yield_((slot + 1) % 2)
        return a
""")
    before = passes.convert_to_ssa()(program)
    before_body = next(iter(before.functions.values())).body
    assert isinstance(before_body, ir.SeqStmts)
    original = next(stmt for stmt in before_body.stmts if isinstance(stmt, ir.ForStmt))
    after = passes.init_mem_ref()(before)
    after_body = next(iter(after.functions.values())).body
    assert isinstance(after_body, ir.SeqStmts)
    loop = next(stmt for stmt in after_body.stmts if isinstance(stmt, ir.ForStmt))
    assert isinstance(loop.body, ir.SeqStmts)
    load = next(
        stmt
        for stmt in loop.body.stmts
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.var.type, ir.TileType)
    )
    assert isinstance(load.var.type, ir.TileType)
    memref = load.var.type.memref
    assert memref is not None and memref.slot_index_ is not None
    assert not memref.is_pinned_
    assert loop.iter_args[0].unique_id != original.iter_args[0].unique_id
    assert isinstance(memref.slot_index_, ir.IterArg)
    assert memref.slot_index_.unique_id == loop.iter_args[0].unique_id
    assert isinstance(memref.byte_offset_, ir.Mul)
    assert isinstance(memref.byte_offset_.left, ir.IterArg)
    assert memref.byte_offset_.left.unique_id == loop.iter_args[0].unique_id
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.UseAfterDef)
    errors = [
        diagnostic.message
        for diagnostic in passes.PropertyVerifierRegistry.verify(properties, after)
        if diagnostic.severity == passes.DiagnosticSeverity.Error
    ]
    assert not errors


def test_repeated_inner_loop_preserves_invariant_seed():
    """One lexical seed use can still observe its old value on repeated entry."""
    _, lowered = _lower(_program(repeat=True))
    assert _StorageInterpreter(lowered, 7).run() == [0, 1, 2, 3, 4, 5, 6, 6] * 2


def test_user_declared_seed_keeps_its_allocation():
    before, after = _lower(_program(declared_seed=True))
    ir.assert_structural_equal(after, before)


def test_rotation_cannot_escape_into_an_unconverted_carry():
    program = pl.parse_program(f"""
@pl.program
class EscapingRotation:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[512, 64], pl.FP16], old_seed: {_PANEL}) -> pl.Tensor[[512, 64], pl.FP16]:
        seed: {_PANEL} = pl.tile.load(a, [0, 0], [64, 64], target_memory=pl.Mem.Mat)
        for block, (current, lag) in pl.range(7, init_values=(seed, old_seed)):
            following: {_PANEL} = pl.tile.load(a, [(block + 1) * 64, 0], [64, 64], target_memory=pl.Mem.Mat)
            consumed: pl.Tile[[64, 64], pl.FP16, pl.Mem.Left] = pl.tile.move(
                current, target_memory=pl.Mem.Left)
            final, saved = pl.yield_(following, current)
        return a
""")
    before, after = _lower(program)
    ir.assert_structural_equal(after, before)


@pytest.mark.parametrize("through_view", [False, True])
def test_loop_result_cannot_escape_into_an_enclosing_carry(through_view):
    """An outer carry can keep an inner bank live across the next inner entry."""
    result = "pl.tile.reshape(final, [64, 64])" if through_view else "final"
    program = pl.parse_program(f"""
@pl.program
class EscapingResult:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[512, 64], pl.FP16], old_seed: {_PANEL}) -> pl.Tensor[[512, 64], pl.FP16]:
        seed: {_PANEL} = pl.tile.load(a, [0, 0], [64, 64], target_memory=pl.Mem.Mat)
        for outer, (saved,) in pl.range(2, init_values=(old_seed,)):
            for block, (current,) in pl.range(7, init_values=(seed,)):
                following: {_PANEL} = pl.tile.load(
                    a, [(block + 1) * 64, 0], [64, 64], target_memory=pl.Mem.Mat)
                consumed: pl.Tile[[64, 64], pl.FP16, pl.Mem.Left] = pl.tile.move(
                    current, target_memory=pl.Mem.Left)
                final = pl.yield_(following)
            selected: {_PANEL} = {result}
            last = pl.yield_(selected)
        return a
""")
    before, after = _lower(program)
    ir.assert_structural_equal(after, before)


@pytest.mark.parametrize("consume_first", [False, True])
def test_copy_free_single_bank_carries_are_not_expanded(consume_first):
    """An unused old value or an already-consumed old value needs just one bank."""
    consume = (
        "consumed: pl.Tile[[64, 64], pl.FP16, pl.Mem.Left] = pl.tile.move(current, target_memory=pl.Mem.Left)"
        if consume_first
        else ""
    )
    program = pl.parse_program(f"""
@pl.program
class SingleBank:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[512, 64], pl.FP16]) -> pl.Tensor[[512, 64], pl.FP16]:
        seed: {_PANEL} = pl.tile.load(a, [0, 0], [64, 64], target_memory=pl.Mem.Mat)
        for block, (current,) in pl.range(7, init_values=(seed,)):
            {consume}
            following: {_PANEL} = pl.tile.load(a, [(block + 1) * 64, 0], [64, 64], target_memory=pl.Mem.Mat)
            final = pl.yield_(following)
        return a
""")
    before, after = _lower(program)
    ir.assert_structural_equal(after, before)


def test_multiple_live_prefetch_leaves_get_distinct_slots():
    """Two fresh panels plus the old current require three genuinely live banks."""
    program = pl.parse_program(f"""
@pl.program
class MultipleLeaves:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[2048, 64], pl.FP16],
               stop: pl.Scalar[pl.INDEX]) -> pl.Tensor[[2048, 64], pl.FP16]:
        seed: {_PANEL} = pl.tile.load(a, [0, 0], [64, 64], target_memory=pl.Mem.Mat)
        for block, (current,) in pl.range(stop, init_values=(seed,)):
            first: {_PANEL} = pl.tile.load(
                a, [(block + 1) * 64, 0], [64, 64], target_memory=pl.Mem.Mat)
            second: {_PANEL} = pl.tile.load(
                a, [(block + 2) * 64, 0], [64, 64], target_memory=pl.Mem.Mat)
            if block % 2 == 0:
                following: {_PANEL} = first
            else:
                following: {_PANEL} = second
            consumed: pl.Tile[[64, 64], pl.FP16, pl.Mem.Left] = pl.tile.move(
                current, target_memory=pl.Mem.Left)
            final = pl.yield_(following)
        final_read: pl.Tile[[64, 64], pl.FP16, pl.Mem.Left] = pl.tile.move(
            final, target_memory=pl.Mem.Left)
        return a
""")
    _, lowered = _lower(program)
    interpreter = _StorageInterpreter(lowered, 7)
    assert interpreter.run() == [0, 1, 3, 3, 5, 5, 7, 7]
    assert len({key for key, _ in interpreter.writes}) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
