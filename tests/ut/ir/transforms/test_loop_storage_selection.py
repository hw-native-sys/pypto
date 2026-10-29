# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compile Mat loop storage choices using INDEX carries and independent slots.

These CPU tests validate lowering and PTOAS acceptance. They do not execute the
loops or establish numerical correctness, slot-reuse synchronization, or timing.
"""

import re
from pathlib import Path

import pypto.language as pl
import pytest
from pypto import DataType, ir, passes
from pypto.arith import Analyzer
from pypto.backend import BackendType
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.runtime import ExecutionMode

from .test_mat_control_flow_transfers import entry as original_rotation_entry

pytestmark = [pytest.mark.usefixtures("ascend_backend"), pytest.mark.ptoas_compile]


def _while_rotation_program(trips: int) -> ir.Program:
    @pl.program
    class WhileRotation:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[512, 64], pl.FP16],
            b: pl.Tensor[[64, 16], pl.FP16],
            out: pl.Out[pl.Tensor[[576, 16], pl.FP32]],
        ) -> pl.Tensor[[576, 16], pl.FP32]:
            initial = pl.load(a, [0, 0], [64, 64], target_memory=pl.Mem.Mat)
            weight = pl.load(b, [0, 0], [64, 16], target_memory=pl.Mem.Mat)
            right = pl.move(weight, target_memory=pl.Mem.Right)
            for current, block in pl.while_(init_values=(initial, pl.const(0, pl.INDEX))):
                pl.cond(block < trips)
                if block + 1 < trips:
                    following = pl.load(a, [(block + 1) * 64, 0], [64, 64], target_memory=pl.Mem.Mat)
                else:
                    following = current
                left = pl.move(current, target_memory=pl.Mem.Left)
                result = pl.tile.matmul(left, right)
                pl.store(result, [block * 64, 0], out)
                final_current, _final_block = pl.yield_(following, block + 1)
            # In particular, a zero-trip loop must expose its initial reference.
            # The final iteration aliases current instead of loading past trips.
            final_left = pl.move(final_current, target_memory=pl.Mem.Left)
            final_result = pl.tile.matmul(final_left, right)
            pl.store(final_result, [512, 0], out)
            return out

    return WhileRotation


@pl.program
class DualForRotation:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[256, 64], pl.FP16],
        b: pl.Tensor[[256, 16], pl.FP16],
        out: pl.Out[pl.Tensor[[320, 16], pl.FP32]],
    ) -> pl.Tensor[[320, 16], pl.FP32]:
        current_a = pl.load(a, [0, 0], [64, 64], target_memory=pl.Mem.Mat)
        current_b = pl.load(b, [0, 0], [64, 16], target_memory=pl.Mem.Mat)
        for block in pl.range(4):
            if block + 1 < 4:
                next_a = pl.load(a, [(block + 1) * 64, 0], [64, 64], target_memory=pl.Mem.Mat)
                next_b = pl.load(b, [(block + 1) * 64, 0], [64, 16], target_memory=pl.Mem.Mat)
            else:
                next_a = current_a
                next_b = current_b
            left = pl.move(current_a, target_memory=pl.Mem.Left)
            right = pl.move(current_b, target_memory=pl.Mem.Right)
            result = pl.tile.matmul(left, right)
            pl.store(result, [block * 64, 0], out)
            current_a = next_a
            current_b = next_b
        final_left = pl.move(current_a, target_memory=pl.Mem.Left)
        final_right = pl.move(current_b, target_memory=pl.Mem.Right)
        final_result = pl.tile.matmul(final_left, final_right)
        pl.store(final_result, [256, 0], out)
        return out


@pl.program
class DualWhileRotation:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[256, 64], pl.FP16],
        b: pl.Tensor[[256, 16], pl.FP16],
        out: pl.Out[pl.Tensor[[320, 16], pl.FP32]],
    ) -> pl.Tensor[[320, 16], pl.FP32]:
        initial_a = pl.load(a, [0, 0], [64, 64], target_memory=pl.Mem.Mat)
        initial_b = pl.load(b, [0, 0], [64, 16], target_memory=pl.Mem.Mat)
        for current_a, current_b, block in pl.while_(
            init_values=(initial_a, initial_b, pl.const(0, pl.INDEX))
        ):
            pl.cond(block < 4)
            if block + 1 < 4:
                next_a = pl.load(a, [(block + 1) * 64, 0], [64, 64], target_memory=pl.Mem.Mat)
                next_b = pl.load(b, [(block + 1) * 64, 0], [64, 16], target_memory=pl.Mem.Mat)
            else:
                next_a = current_a
                next_b = current_b
            left = pl.move(current_a, target_memory=pl.Mem.Left)
            right = pl.move(current_b, target_memory=pl.Mem.Right)
            result = pl.tile.matmul(left, right)
            pl.store(result, [block * 64, 0], out)
            final_a, final_b, _final_block = pl.yield_(next_a, next_b, block + 1)
        final_left = pl.move(final_a, target_memory=pl.Mem.Left)
        final_right = pl.move(final_b, target_memory=pl.Mem.Right)
        final_result = pl.tile.matmul(final_left, final_right)
        pl.store(final_result, [256, 0], out)
        return out


@pl.program
class ForSwap:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        pool: pl.Tensor[[32, 32], pl.FP16],
        weight: pl.Tensor[[32, 16], pl.FP16],
        out: pl.Out[pl.Tensor[[64, 16], pl.FP32]],
    ) -> pl.Tensor[[64, 16], pl.FP32]:
        first = pl.load(pool, [0, 0], [16, 32], target_memory=pl.Mem.Mat)
        second = pl.load(pool, [16, 0], [16, 32], target_memory=pl.Mem.Mat)
        for _step, (current, previous) in pl.range(3, init_values=(first, second)):
            final_current, final_previous = pl.yield_(previous, current)
        rhs = pl.load(weight, [0, 0], [32, 16], target_memory=pl.Mem.Mat)
        right = pl.move(rhs, target_memory=pl.Mem.Right)
        left_current = pl.move(final_current, target_memory=pl.Mem.Left)
        result_current = pl.tile.matmul(left_current, right)
        pl.store(result_current, [0, 0], out)
        left_previous = pl.move(final_previous, target_memory=pl.Mem.Left)
        result_previous = pl.tile.matmul(left_previous, right)
        pl.store(result_previous, [16, 0], out)
        # Both original descriptors also remain independently observable.
        left_first = pl.move(first, target_memory=pl.Mem.Left)
        result_first = pl.tile.matmul(left_first, right)
        pl.store(result_first, [32, 0], out)
        left_second = pl.move(second, target_memory=pl.Mem.Left)
        result_second = pl.tile.matmul(left_second, right)
        pl.store(result_second, [48, 0], out)
        return out


@pl.program
class WhileSwap:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        pool: pl.Tensor[[32, 32], pl.FP16],
        weight: pl.Tensor[[32, 16], pl.FP16],
        out: pl.Out[pl.Tensor[[64, 16], pl.FP32]],
    ) -> pl.Tensor[[64, 16], pl.FP32]:
        first = pl.load(pool, [0, 0], [16, 32], target_memory=pl.Mem.Mat)
        second = pl.load(pool, [16, 0], [16, 32], target_memory=pl.Mem.Mat)
        for current, previous, step in pl.while_(init_values=(first, second, pl.const(0, pl.INDEX))):
            pl.cond(step < 3)
            final_current, final_previous, _final_step = pl.yield_(previous, current, step + 1)
        rhs = pl.load(weight, [0, 0], [32, 16], target_memory=pl.Mem.Mat)
        right = pl.move(rhs, target_memory=pl.Mem.Right)
        left_current = pl.move(final_current, target_memory=pl.Mem.Left)
        result_current = pl.tile.matmul(left_current, right)
        pl.store(result_current, [0, 0], out)
        left_previous = pl.move(final_previous, target_memory=pl.Mem.Left)
        result_previous = pl.tile.matmul(left_previous, right)
        pl.store(result_previous, [16, 0], out)
        left_first = pl.move(first, target_memory=pl.Mem.Left)
        result_first = pl.tile.matmul(left_first, right)
        pl.store(result_first, [32, 0], out)
        left_second = pl.move(second, target_memory=pl.Mem.Left)
        result_second = pl.tile.matmul(left_second, right)
        pl.store(result_second, [48, 0], out)
        return out


@pl.program
class WhileShiftRegister:
    """Two carries share one family and protect both seeds past the loop."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        pool: pl.Tensor[[96, 32], pl.FP16],
        weight: pl.Tensor[[32, 16], pl.FP16],
        out: pl.Out[pl.Tensor[[192, 16], pl.FP32]],
    ) -> pl.Tensor[[192, 16], pl.FP32]:
        seed_older = pl.load(pool, [0, 0], [16, 32], target_memory=pl.Mem.Mat)
        seed_current = pl.load(pool, [16, 0], [16, 32], target_memory=pl.Mem.Mat)
        rhs = pl.load(weight, [0, 0], [32, 16], target_memory=pl.Mem.Mat)
        right = pl.move(rhs, target_memory=pl.Mem.Right)
        for older, current, step in pl.while_(init_values=(seed_older, seed_current, pl.const(0, pl.INDEX))):
            pl.cond(step < 4)
            if step + 1 < 4:
                following = pl.load(pool, [(step + 2) * 16, 0], [16, 32], target_memory=pl.Mem.Mat)
            else:
                following = current
            # The repeated producer must avoid both incoming storage choices.
            left_older = pl.move(older, target_memory=pl.Mem.Left)
            result_older = pl.tile.matmul(left_older, right)
            pl.store(result_older, [step * 32, 0], out)
            left_current = pl.move(current, target_memory=pl.Mem.Left)
            result_current = pl.tile.matmul(left_current, right)
            pl.store(result_current, [step * 32 + 16, 0], out)
            final_older, final_current, _final_step = pl.yield_(current, following, step + 1)
        left_seed_older = pl.move(seed_older, target_memory=pl.Mem.Left)
        result_seed_older = pl.tile.matmul(left_seed_older, right)
        pl.store(result_seed_older, [128, 0], out)
        left_seed_current = pl.move(seed_current, target_memory=pl.Mem.Left)
        result_seed_current = pl.tile.matmul(left_seed_current, right)
        pl.store(result_seed_current, [144, 0], out)
        left_final_older = pl.move(final_older, target_memory=pl.Mem.Left)
        result_final_older = pl.tile.matmul(left_final_older, right)
        pl.store(result_final_older, [160, 0], out)
        left_final_current = pl.move(final_current, target_memory=pl.Mem.Left)
        result_final_current = pl.tile.matmul(left_final_current, right)
        pl.store(result_final_current, [176, 0], out)
        return out


@pl.program
class PinnedMatWhile:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        source: pl.Tensor[[32, 32], pl.FP16],
        out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        seed: pl.Tile[[16, 32], pl.FP16, pl.MemRef("fixed", slots=2)[0], pl.Mem.Mat] = pl.load(
            source, [0, 0], [16, 32], target_memory=pl.Mem.Mat
        )
        replacement: pl.Tile[[16, 32], pl.FP16, pl.MemRef("fixed", slots=2)[1], pl.Mem.Mat] = pl.load(
            source, [16, 0], [16, 32], target_memory=pl.Mem.Mat
        )
        for current, step in pl.while_(init_values=(seed, pl.const(0, pl.INDEX))):
            pl.cond(step < 3)
            _consume = pl.move(current, target_memory=pl.Mem.Left)
            final, _final_step = pl.yield_(replacement, step + 1)
        _consume_final = pl.move(final, target_memory=pl.Mem.Left)
        return out


@pl.program
class PinnedVecWhile:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        source: pl.Tensor[[32, 32], pl.FP32],
        out: pl.Out[pl.Tensor[[48, 32], pl.FP32]],
    ) -> pl.Tensor[[48, 32], pl.FP32]:
        seed: pl.Tile[[16, 32], pl.FP32, pl.MemRef("fixed", slots=2)[0], pl.Mem.Vec] = pl.load(
            source, [0, 0], [16, 32], target_memory=pl.Mem.Vec
        )
        replacement: pl.Tile[[16, 32], pl.FP32, pl.MemRef("fixed", slots=2)[1], pl.Mem.Vec] = pl.load(
            source, [16, 0], [16, 32], target_memory=pl.Mem.Vec
        )
        for current, step in pl.while_(init_values=(seed, pl.const(0, pl.INDEX))):
            pl.cond(step < 3)
            pl.store(current, [0, 0], out)
            final, _final_step = pl.yield_(replacement, step + 1)
        pl.store(final, [16, 0], out)
        # The fixed carry must not overwrite the independently observed seed.
        pl.store(seed, [32, 0], out)
        return out


def _case(scenario: str) -> tuple[ir.Program, type, int, dict[tuple[int, int], int]]:
    """Return the program, user loop kind, minimum carry count and live slots."""
    if scenario == "original_3039":
        return original_rotation_entry.specialize(), ir.ForStmt, 1, {(64, 64): 2}
    if scenario.startswith("while_"):
        trips = int(scenario.removeprefix("while_"))
        return (
            _while_rotation_program(trips),
            ir.WhileStmt,
            2 if trips > 1 else 0,
            {(64, 64): 2 if trips > 1 else 1},
        )
    if scenario == "dual_for":
        return DualForRotation, ir.ForStmt, 2, {(64, 64): 2, (64, 16): 2}
    if scenario == "dual_while":
        return DualWhileRotation, ir.WhileStmt, 3, {(64, 64): 2, (64, 16): 2}
    if scenario == "swap_for":
        return ForSwap, ir.ForStmt, 2, {(16, 32): 2}
    if scenario == "shift_while":
        # Two independently observed seeds plus at least one free producer slot
        # for each possible occupied carry require the five-slot family here.
        return WhileShiftRegister, ir.WhileStmt, 3, {(16, 32): 5}
    assert scenario == "swap_while"
    return WhileSwap, ir.WhileStmt, 3, {(16, 32): 2}


def _loops(program: ir.Program) -> list[ir.ForStmt | ir.WhileStmt]:
    result: list[ir.ForStmt | ir.WhileStmt] = []

    class Collect(ir.IRVisitor):
        def visit_for_stmt(self, op: ir.ForStmt) -> None:
            result.append(op)
            super().visit_for_stmt(op)

        def visit_while_stmt(self, op: ir.WhileStmt) -> None:
            result.append(op)
            super().visit_while_stmt(op)

    Collect().visit_program(program)
    return result


def _assert_index_carries(program: ir.Program, kind: type, minimum: int) -> None:
    loops = _loops(program)
    for loop in loops:
        for argument in loop.iter_args:
            assert isinstance(argument.type, ir.ScalarType), argument.name_hint
            assert argument.type.dtype == DataType.INDEX, argument.name_hint
    candidates = [loop for loop in loops if isinstance(loop, kind)]
    if minimum:
        assert candidates, ir.python_print(program)
        assert max(len(loop.iter_args) for loop in candidates) >= minimum


def _assert_slot_geometry(program: ir.Program, required: dict[tuple[int, int], int]) -> None:
    pools: dict[tuple[int, int], dict[int, int]] = {shape: {} for shape in required}
    allocation_sizes: dict[int, int] = {}
    required_sizes: dict[int, int] = {}
    analyzer = Analyzer()

    class Collect(ir.IRVisitor):
        def visit_assign_stmt(self, op: ir.AssignStmt) -> None:
            if isinstance(op.value, ir.Call) and op.value.op.name == ir.get_op("tile.alloc").name:
                assert isinstance(op.value.args[1], ir.ConstInt)
                allocation_sizes[op.var.unique_id] = op.value.args[1].value
            tile = op.var.type
            if isinstance(tile, ir.TileType) and tile.memory_space == ir.MemorySpace.Mat and tile.memref:
                dimensions: list[int] = []
                for extent in tile.shape:
                    assert isinstance(extent, ir.ConstInt)
                    dimensions.append(extent.value)
                assert len(dimensions) == 2
                shape = (dimensions[0], dimensions[1])
                if shape in pools:
                    memory = tile.memref
                    assert memory.size_ > 0 and memory.slot_count_ >= 1
                    if memory.slot_count_ > 1:
                        assert memory.slot_index_ is not None
                        assert isinstance(memory.slot_index_.type, ir.ScalarType)
                        assert memory.slot_index_.type.dtype == DataType.INDEX
                        # Before address assignment, slot k starts at k * stride.
                        # The allocation reserves the full interval for all slots.
                        expected_offset = ir.Mul(
                            memory.slot_index_,
                            ir.ConstInt(memory.size_, DataType.INDEX, op.span),
                            DataType.INDEX,
                            op.span,
                        )
                        assert ir.structural_equal(
                            analyzer.simplify(memory.byte_offset_), analyzer.simplify(expected_offset)
                        )
                    required_sizes[memory.base_.unique_id] = max(
                        required_sizes.get(memory.base_.unique_id, 0), memory.size_ * memory.slot_count_
                    )
                    pools[shape][memory.base_.unique_id] = max(
                        pools[shape].get(memory.base_.unique_id, 0), memory.slot_count_
                    )
            super().visit_assign_stmt(op)

    Collect().visit_program(program)
    for shape, minimum in required.items():
        assert sum(pools[shape].values()) >= minimum, (shape, pools[shape])
    for base, minimum in required_sizes.items():
        assert allocation_sizes[base] >= minimum, (base, allocation_sizes[base], minimum)


def _assert_parallel_swap(program: ir.Program, kind: type) -> None:
    loop = next(loop for loop in _loops(program) if isinstance(loop, kind) and len(loop.iter_args) >= 2)
    statements = loop.body.stmts if isinstance(loop.body, ir.SeqStmts) else [loop.body]
    yielded = next(statement for statement in reversed(statements) if isinstance(statement, ir.YieldStmt))
    assert yielded.value[0].same_as(loop.iter_args[1])
    assert yielded.value[1].same_as(loop.iter_args[0])


def _assert_pto_slots(text: str, planner: passes.MemoryPlanner, required: dict[tuple[int, int], int]) -> None:
    constants = dict(re.findall(r"(%\w+) = arith\.constant (\d+) : i64", text))
    windows: dict[tuple[int, int], set[int | str]] = {shape: set() for shape in required}
    runtime_windows: set[tuple[int, int]] = set()
    for line in text.splitlines():
        if "pto.tmov" in line:
            assert len(re.findall(r"loc\s*=\s*mat(?:,|>)", line)) < 2, line
        if ("scf.for" in line and "iter_args" in line) or "scf.while" in line:
            assert "!pto.tile_buf" not in line, line
        if "pto.alloc_tile" not in line or "loc=mat" not in line:
            continue
        dimensions = re.search(r"rows=(\d+), cols=(\d+)", line)
        assert dimensions is not None, line
        shape = (int(dimensions.group(1)), int(dimensions.group(2)))
        if shape not in windows:
            continue
        address = re.search(r"addr = (%\w+)", line)
        if planner == passes.MemoryPlanner.PTOAS:
            # Independent alloc roots are visible to PTOAS; selected aliases
            # must not be mistaken for additional physical slots.
            if address is None:
                allocation = re.search(r"(%\w+) = pto\.alloc_tile", line)
                assert allocation is not None
                windows[shape].add(allocation.group(1))
        elif address and address.group(1) in constants:
            windows[shape].add(int(constants[address.group(1)]))
        elif address:
            runtime_windows.add(shape)
    for shape, minimum in required.items():
        if planner == passes.MemoryPlanner.PTOAS:
            assert len(windows[shape]) >= minimum, (shape, windows[shape])
        else:
            # Addressed descriptors may choose slots using runtime byte offsets;
            # their disjoint stride and reserved capacity are checked in IR above.
            assert len(windows[shape]) >= minimum or shape in runtime_windows, (shape, windows[shape])
            offsets: list[int] = []
            for address_value in windows[shape]:
                assert isinstance(address_value, int)
                offsets.append(address_value)
            offsets.sort()
            byte_count = shape[0] * shape[1] * 2  # All exercised Mat streams are FP16.
            assert all(right - left >= byte_count for left, right in zip(offsets, offsets[1:]))


@pytest.mark.parametrize(
    "scenario",
    [
        "original_3039",
        "while_0",
        "while_1",
        "while_3",
        "while_4",
        "dual_for",
        "dual_while",
        "swap_for",
        "swap_while",
        "shift_while",
    ],
)
@pytest.mark.parametrize("planner", list(passes.MemoryPlanner))
@pytest.mark.parametrize("enable_buffer_ir", [False, True], ids=["tile", "buffer_requested"])
def test_loop_storage_selection_compiles(
    tmp_path: Path,
    pass_verification_instruments,
    scenario: str,
    planner: passes.MemoryPlanner,
    enable_buffer_ir: bool,
):
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is unavailable")
    program, loop_kind, minimum_carries, required_slots = _case(scenario)
    snapshots: dict[str, ir.Program] = {}

    def capture_after(current_pass: passes.Pass, current_program: ir.Program) -> None:
        snapshots["last"] = current_program
        if current_pass.get_name() == "MaterializeSemanticAliases":
            snapshots["storage"] = current_program

    capture = passes.CallbackInstrument(after_pass=capture_after)
    with passes.PassContext(
        [*pass_verification_instruments, capture],
        memory_planner=planner,
        enable_buffer_ir=enable_buffer_ir,
    ):
        compiled = ir.compile(
            program,
            output_dir=str(tmp_path),
            backend_type=BackendType.Ascend910,
            execution_mode=ExecutionMode.ONBOARD,
            dump_passes=True,
        )
    assert compiled.output_dir == tmp_path
    assert list((tmp_path / "kernels").rglob("*.cpp"))
    assert list((tmp_path / "passes_dump").glob("*_after_MaterializeSemanticAliases.py"))
    _assert_index_carries(snapshots["last"], loop_kind, minimum_carries)
    _assert_slot_geometry(snapshots["storage"], required_slots)
    if scenario.startswith("swap_"):
        _assert_parallel_swap(snapshots["storage"], loop_kind)
    sources = list((tmp_path / "ptoas").glob("*.pto"))
    assert sources
    text = "\n".join(source.read_text() for source in sources)
    _assert_pto_slots(text, planner, required_slots)


@pytest.mark.parametrize("planner", list(passes.MemoryPlanner))
@pytest.mark.parametrize("enable_buffer_ir", [False, True], ids=["tile", "buffer_requested"])
def test_pinned_mat_while_rejects_unsupported_backedge_copy(
    tmp_path: Path, pass_verification_instruments, planner: passes.MemoryPlanner, enable_buffer_ir: bool
):
    with passes.PassContext(
        pass_verification_instruments, memory_planner=planner, enable_buffer_ir=enable_buffer_ir
    ):
        with pytest.raises(
            ValueError,
            match=(
                r"memory-space pair is unsupported|unsupported tile.move|"
                r"cannot preserve an independently live loop input without a same-space copy"
            ),
        ):
            ir.compile(
                PinnedMatWhile,
                output_dir=str(tmp_path),
                backend_type=BackendType.Ascend910,
                execution_mode=ExecutionMode.ONBOARD,
                skip_ptoas=True,
                dump_passes=False,
            )


@pytest.mark.parametrize("planner", list(passes.MemoryPlanner))
@pytest.mark.parametrize("enable_buffer_ir", [False, True], ids=["tile", "buffer_requested"])
def test_pinned_vec_while_materializes_legal_backedge_copy(
    tmp_path: Path, pass_verification_instruments, planner: passes.MemoryPlanner, enable_buffer_ir: bool
):
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is unavailable")
    snapshots: dict[str, ir.Program] = {}

    def capture_after(current_pass: passes.Pass, current_program: ir.Program) -> None:
        if current_pass.get_name() in {"MaterializeSemanticAliases", "MemoryReuse"}:
            snapshots["storage"] = current_program

    capture = passes.CallbackInstrument(after_pass=capture_after)
    with passes.PassContext(
        [*pass_verification_instruments, capture], memory_planner=planner, enable_buffer_ir=enable_buffer_ir
    ):
        ir.compile(
            PinnedVecWhile,
            output_dir=str(tmp_path),
            backend_type=BackendType.Ascend910,
            execution_mode=ExecutionMode.ONBOARD,
            dump_passes=True,
        )
    loop = next(loop for loop in _loops(snapshots["storage"]) if isinstance(loop, ir.WhileStmt))
    assert isinstance(loop.iter_args[0].type, ir.TileType)
    statements = loop.body.stmts if isinstance(loop.body, ir.SeqStmts) else [loop.body]
    copies = [
        statement
        for statement in statements
        if isinstance(statement, ir.AssignStmt)
        and isinstance(statement.value, ir.Call)
        and statement.value.op.name == ir.get_op("tile.move").name
    ]
    assert len(copies) == 1, ir.python_print(snapshots["storage"])
    copy = copies[0].value
    assert isinstance(copy, ir.Call)
    source_type, destination_type = copy.args[0].type, copies[0].var.type
    assert isinstance(source_type, ir.TileType) and isinstance(destination_type, ir.TileType)
    assert source_type.memory_space == destination_type.memory_space == ir.MemorySpace.Vec
    assert source_type.memref is not None and destination_type.memref is not None
    assert not ir.structural_equal(source_type.memref.byte_offset_, destination_type.memref.byte_offset_)
    carry_memref = loop.iter_args[0].type.memref
    assert carry_memref is not None
    assert ir.structural_equal(destination_type.memref, carry_memref)
    assignments: list[ir.AssignStmt] = []

    class Assignments(ir.IRVisitor):
        def visit_assign_stmt(self, op: ir.AssignStmt) -> None:
            assignments.append(op)
            super().visit_assign_stmt(op)

    Assignments().visit_program(snapshots["storage"])
    seed = next(statement for statement in assignments if statement.var.name_hint.startswith("seed"))
    seed_type = seed.var.type
    assert isinstance(seed_type, ir.TileType) and seed_type.memref is not None
    assert not carry_memref.base_.same_as(seed_type.memref.base_)
    entry = next(statement for statement in assignments if statement.var.same_as(loop.iter_args[0].initValue))
    assert isinstance(entry.value, ir.Call) and entry.value.op.name == ir.get_op("tile.move").name
    assert entry.value.args[0].same_as(seed.var)
    yielded = next(statement for statement in reversed(statements) if isinstance(statement, ir.YieldStmt))
    assert yielded.value[0].same_as(copies[0].var)
    sources = list((tmp_path / "ptoas").glob("*.pto"))
    assert sources
    assert any(
        "pto.tmov" in line and len(re.findall(r"loc\s*=\s*vec(?:,|>)", line)) == 2
        for source in sources
        for line in source.read_text().splitlines()
    )
    assert list((tmp_path / "kernels").rglob("*.cpp"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
